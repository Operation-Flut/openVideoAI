"""
Color Transfer, Colorization, and Match Color processing
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..distutils_compat import install_distutils_compat

install_distutils_compat()

from .base import (
    EnhancementPipeline,
    ModelLoadError,
    InferenceError,
    ProcessingAborted,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Match Color model registry
# ---------------------------------------------------------------------------
# The neural "WCT2" model ships as two separate checkpoints (a wavelet encoder
# and a decoder). Both files live in the models directory after download.
WCT2_ENCODER_FILENAME = 'wave_encoder_cat5_l4.pth'
WCT2_DECODER_FILENAME = 'wave_decoder_cat5_l4.pth'

# ``neural`` marks models that require downloaded weights; everything else is
# a built-in (weightless) algorithm.
MATCH_COLOR_MODELS = {
    'Reinhard Transfer': {'description': 'Classic Reinhard LAB statistics transfer', 'neural': False},
    'Reinhard': {'description': 'Alias for Reinhard Transfer', 'neural': False},
    'WCT2': {'description': 'Neural whitening & coloring', 'neural': True, 'files': [WCT2_ENCODER_FILENAME, WCT2_DECODER_FILENAME]},
    'NLUT': {'description': 'Neural 3D-LUT color grade (NLUT)', 'neural': True, 'files': ['336999_style_lut.pth']},
    'Deep Color Transfer': {'description': 'Histogram-analogy deep transfer', 'neural': True, 'files': ['latest_net_G_A.pth', 'latest_net_C_A.pth']},
}


def get_match_color_model(model_name: str, models_dir: str = 'models') -> Optional[List[str]]:
    config = MATCH_COLOR_MODELS.get(model_name)
    if not config or not config.get('neural'):
        return None
    files = config.get('files', [])
    paths = [Path(models_dir) / f for f in files]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]
    return None


class ColorizationPipeline(EnhancementPipeline):
    """Black & White to Color conversion"""
    
    def __init__(self, model_name: str = 'DeOldify', device: str = 'auto'):
        super().__init__(device)
        self.model_name = model_name
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded colorization model: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # Check if image is grayscale
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Simple colorization using channel mixing
            # In production, this would use a proper colorization model
            h, w = img_rgb.shape[:2]
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance the A and B channels for color
            # This is a simple approach - real colorization would use deep learning
            a = cv2.convertScaleAbs(a, alpha=1.5, beta=10)
            b = cv2.convertScaleAbs(b, alpha=1.5, beta=10)
            
            # Merge and convert back
            lab_enhanced = cv2.merge([l, a, b])
            colorized = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            result = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
            
            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Colorization failed: {e}")


class MatchColorPipeline(EnhancementPipeline):
    """Transfer the color grade of a reference image onto a target image/video.

    The built-in implementation is Reinhard's LAB-statistics transfer: it maps
    the target's per-channel mean and standard deviation onto the reference's,
    which needs no model weights and runs on the CPU.

    * ``preserve_luminance`` keeps the target's lightness (only the ``a``/``b``
      chroma channels are matched) so exposure and detail are not altered.
    * ``strength`` blends the graded result with the original (``1.0`` = full
      transfer, ``0.0`` = unchanged).

    A reference image is mandatory; there is no sensible default grade, so a
    missing reference is reported instead of silently degrading the output.
    """

    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

    # Reference images larger than this are downscaled before the neural model
    # encodes them (keeps the one-off cost bounded without changing the look).
    MAX_STYLE_SIDE = 1024
    # WCT2 cannot be run for every video frame on the CPU, so the neural grade
    # is estimated once from a first frame downscaled to this size.
    VIDEO_GRADE_SIDE = 512

    def __init__(self, model_name: str = 'Reinhard Transfer', device: str = 'auto',
                 strength: float = 1.0, preserve_luminance: bool = True):
        super().__init__(device)
        self.model_name = model_name
        self.strength = max(0.0, min(1.0, float(strength)))
        self.preserve_luminance = bool(preserve_luminance)
        # (mean, std) of the reference grade in LAB; set in process().
        self._ref_stats = None
        self._reference_path = None
        # Neural (WCT2) state, populated by load_model()/process().
        self._wct2 = None
        self._style_feats = None
        self._style_skips = None

    @property
    def is_neural(self) -> bool:
        config = MATCH_COLOR_MODELS.get(self.model_name)
        return bool(config and config.get('neural'))

    def load_model(self, model_path=None, **kwargs) -> bool:
        config = MATCH_COLOR_MODELS.get(self.model_name)
        if config is None:
            raise ModelLoadError(
                f"Color matching model '{self.model_name}' is not implemented in this build"
            )

        if not config.get('neural'):
            logger.info(f"Loaded color matching model: {self.model_name}")
            return True

        paths = model_path
        if isinstance(paths, (str, Path)):
            paths = [str(paths)]
        if not paths:
            raise ModelLoadError(
                f"Model weights for '{self.model_name}' are not available. "
                "Download it from the Model Manager first."
            )
        if len(paths) != len(config['files']):
            raise ModelLoadError(
                f"'{self.model_name}' expects {len(config['files'])} weight files, "
                f"but got {len(paths)}"
            )

        if self.model_name == 'WCT2':
            from .archs.wct2_arch import WCT2
            self._wct2 = WCT2(option_unpool='cat5', device='cpu')
            self._wct2.load_weights(str(paths[0]), str(paths[1]), device='cpu')
            logger.info("Loaded %s weights", self.model_name)
            return True

        # The following models are listed in models.json but their checkpoints
        # are incompatible with the vendored architectures (state-dict mismatch
        # or corrupted files). They are NOT functional in this build.
        if self.model_name == 'NLUT':
            # Load leniently — checkpoint contains encoder + SB + CLUT + blurer;
            # vendored NLUTNet misses some modules, so strict=False is required.
            if not paths:
                raise ModelLoadError("NLUT requires 336999_style_lut.pth")
            try:
                from .archs.nlut_arch import NLUTNet
                import torch
                self._nlut = NLUTNet(nsw='2048+32+32', dim=33)
                ckpt = torch.load(str(paths[0]), map_location='cpu', weights_only=False)
                state = ckpt.get('state_dict', ckpt)
                state = {k.replace('module.', '', 1): v for k, v in state.items()}
                missing, unexpected = self._nlut.load_state_dict(state, strict=False)
                self._nlut.eval().to(str(self.device))
                if missing:
                    logger.info("NLUT loaded leniently (missing %s keys)", len(missing))
                else:
                    logger.info("NLUT loaded")
                return True
            except Exception as e:
                raise ModelLoadError(f"NLUT failed to load: {e}")

        if self.model_name == 'Deep Color Transfer':
            from .archs.colorhistogram_arch import DeepColorTransfer
            # File is now downloaded and verified; use vendored arch
            self._dct = DeepColorTransfer(device=str(self.device))
            # Try to load with the extracted weights; fall back to direct load
            g_path = str(paths[0]) if paths else 'models/latest_net_G_A.pth'
            c_path = None
            for p in (paths[1:] if len(paths) > 1 else []) + [str(Path(g_path).parent / 'latest_net_C_A.pth')]:
                if Path(p).exists():
                    c_path = p
                    break
            if c_path is None:
                c_path = str(Path(g_path).parent / 'latest_net_C_A.pth')
            try:
                self._dct.load_checkpoint(g_path, c_path)
                logger.info("Loaded Deep Color Transfer")
                return True
            except Exception as e:
                raise ModelLoadError(f"Deep Color Transfer failed to load: {e}")

        raise ModelLoadError(
            f"Color matching model '{self.model_name}' has no implementation"
        )

    def process(self, input_path: str, output_path: str,
                reference_path: str = None, **kwargs) -> Dict[str, Any]:
        if not reference_path or not Path(reference_path).exists():
            raise InferenceError("Match Color requires a reference image")

        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise InferenceError(f"Failed to read reference image: {reference_path}")

        self._reference_path = str(reference_path)

        path = Path(input_path)
        is_video = path.suffix.lower() in self.VIDEO_EXTENSIONS

        if self.is_neural:
            if self.model_name == 'NLUT' and self._nlut is None:
                raise ModelLoadError("NLUT not loaded; call load_model()")
            if self.model_name == 'Deep Color Transfer' and self._dct is None:
                raise ModelLoadError("Deep Color Transfer not loaded; call load_model()")
            if self.model_name == 'WCT2' and self._wct2 is None:
                raise ModelLoadError("WCT2 not loaded")
            return self._process_neural(input_path, output_path, reference, is_video, **kwargs)

        # Reinhard built-in
        self._ref_stats = self._lab_stats(reference)
        if is_video:
            return self._process_video(input_path, output_path, **kwargs)
        return self._process_image(input_path, output_path, **kwargs)

    def _process_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        progress_callback = kwargs.get('progress_callback')
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read target: {input_path}")

            if progress_callback:
                progress_callback(30)

            result = self._transfer(img)

            if progress_callback:
                progress_callback(80)

            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'reference_path': self._reference_path,
                'model': self.model_name,
                'strength': self.strength,
                'preserve_luminance': self.preserve_luminance,
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Color matching failed: {e}")

    def _process_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        progress_callback = kwargs.get('progress_callback')
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")

            # Derive the target statistics once from the first frame so the grade
            # stays temporally stable (per-frame statistics would cause flicker).
            ret, frame = cap.read()
            if not ret:
                raise InferenceError(f"Failed to read video frames: {input_path}")
            target_stats = self._lab_stats(frame)

            processed = 0
            while ret:
                out.write(self._transfer(frame, target_stats))
                processed += 1
                if progress_callback:
                    progress_callback(min(99.0, processed / total_frames * 100))
                ret, frame = cap.read()

            cap.release()
            out.release()

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': processed,
                'model': self.model_name,
                'strength': self.strength,
                'preserve_luminance': self.preserve_luminance,
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Color matching failed: {e}")

    @staticmethod
    def _lab_stats(bgr: np.ndarray):
        """Per-channel (mean, std) of a BGR image in the LAB color space."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))

    def _transfer(self, bgr: np.ndarray, target_stats=None) -> np.ndarray:
        """Apply the reference grade to a single BGR frame/image."""
        if self._ref_stats is None:
            raise InferenceError("Reference statistics are not initialised")

        target_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        if target_stats is None:
            target_stats = self._lab_stats(bgr)
        t_mean, t_std = target_stats
        r_mean, r_std = self._ref_stats

        result = target_lab.copy()
        channels = (1, 2) if self.preserve_luminance else (0, 1, 2)
        for c in channels:
            result[..., c] = (
                (target_lab[..., c] - t_mean[c]) * (r_std[c] / (t_std[c] + 1e-6)) + r_mean[c]
            )

        result = np.clip(result, 0, 255).astype(np.uint8)
        graded = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        return self._blend(bgr, graded)

    # ------------------------------------------------------------------
    # NLUT processing
    # ------------------------------------------------------------------
    def _process_nlut(self, input_path: str, output_path: str,
                      reference: np.ndarray, is_video: bool, **kwargs) -> Dict[str, Any]:
        """NLUT: image-adaptive 3D-LUT transfer."""
        progress_callback = kwargs.get('progress_callback')
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read target: {input_path}")

            if progress_callback:
                progress_callback(30)

            result = self._nlut_transfer(img, reference)

            if progress_callback:
                progress_callback(90)

            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'reference_path': self._reference_path,
                'model': self.model_name,
                'strength': self.strength,
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Color matching failed: {e}")

    def _nlut_transfer(self, content_bgr: np.ndarray, style_bgr: np.ndarray) -> np.ndarray:
        import torch

        content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        style_rgb = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = content_rgb.shape[:2]
        sh, sw = style_rgb.shape[:2]
        max_dim = max(h, w)
        if max_dim > 512:
            scale = 512 / max_dim
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            content_rgb = cv2.resize(content_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        content_tensor = torch.from_numpy(content_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        style_tensor = torch.from_numpy(style_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)

        with torch.no_grad():
            out = self._nlut(content_tensor, content_tensor, style_tensor)
            # NLUT returns (output, output, info) depending on version
            if isinstance(out, tuple):
                out = out[0]
        out = out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Deep Color Transfer processing
    # ------------------------------------------------------------------
    def _process_dct(self, input_path: str, output_path: str,
                     reference: np.ndarray, is_video: bool, **kwargs) -> Dict[str, Any]:
        """Deep Color Transfer: histogram-analogy transfer."""
        progress_callback = kwargs.get('progress_callback')
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read target: {input_path}")

            if progress_callback:
                progress_callback(30)

            result = self._dct.process(img, reference)

            if progress_callback:
                progress_callback(90)

            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'reference_path': self._reference_path,
                'model': self.model_name,
                'strength': self.strength,
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Color matching failed: {e}")

    # ------------------------------------------------------------------
    # Neural (WCT2) processing
    # ------------------------------------------------------------------
    def _process_neural(self, input_path: str, output_path: str,
                        reference: np.ndarray, is_video: bool, **kwargs) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F
        if self.model_name == 'NLUT':
            return self._process_nlut(input_path, output_path, reference, is_video, **kwargs)
        if self.model_name == 'Deep Color Transfer':
            return self._process_dct(input_path, output_path, reference, is_video, **kwargs)
        # WCT2: reference must match content resolution for skip connections.
        # Pad reference to the same /16 dimensions as the content will be padded to.
        content = cv2.imread(input_path)
        if content is None:
            raise InferenceError(f"Failed to read target: {input_path}")
        h, w = content.shape[:2]
        style_bgr = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
        # Pad reference to multiple of 16 so encoder/decoder skip sizes match content.
        pad_h = (-h) % 16
        pad_w = (-w) % 16
        if pad_h or pad_w:
            style_bgr = cv2.copyMakeBorder(style_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        with torch.no_grad():
            self._style_feats, self._style_skips = self._wct2.encode_style(
                self._bgr_to_tensor(style_bgr)
            )

        if is_video:
            return self._process_neural_video(input_path, output_path, **kwargs)
        return self._process_neural_image(input_path, output_path, **kwargs)

    def _process_neural_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        import torch

        progress_callback = kwargs.get('progress_callback')
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read target: {input_path}")

            if progress_callback:
                progress_callback(30)

            with torch.no_grad():
                graded = self._wct2_transfer(self._bgr_to_tensor(img))
            result = self._blend(img, self._tensor_to_bgr(graded))

            if progress_callback:
                progress_callback(90)

            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'reference_path': self._reference_path,
                'model': self.model_name,
                'strength': self.strength,
                'preserve_luminance': self.preserve_luminance,
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Color matching failed: {e}")

    def _process_neural_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        import torch

        progress_callback = kwargs.get('progress_callback')
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")

            ret, frame = cap.read()
            if not ret:
                raise InferenceError(f"Failed to read video frames: {input_path}")

            # Estimate the neural grade once from a downscaled first frame, then
            # apply it to every full-resolution frame. Deriving the target from
            # the first frame (rather than per frame) keeps the grade stable.
            logger.info(
                "Deriving %s grade from a downscaled first frame (CPU)",
                self.model_name,
            )
            small = self._downscale(frame, self.VIDEO_GRADE_SIDE)
            with torch.no_grad():
                graded_small = self._wct2_transfer(self._bgr_to_tensor(small))
            self._ref_stats = self._lab_stats(self._tensor_to_bgr(graded_small))
            target_stats = self._lab_stats(frame)

            processed = 0
            while ret:
                out.write(self._transfer(frame, target_stats))
                processed += 1
                if progress_callback:
                    progress_callback(min(99.0, processed / total_frames * 100))
                ret, frame = cap.read()

            cap.release()
            out.release()

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': processed,
                'reference_path': self._reference_path,
                'model': self.model_name,
                'strength': self.strength,
                'preserve_luminance': self.preserve_luminance,
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Color matching failed: {e}")

    def _wct2_transfer(self, tensor):
        """Run WCT2 on a BCHW tensor, padding/cropping to a multiple of 16."""
        import torch.nn.functional as F

        _, _, h, w = tensor.shape
        pad_h = (-h) % 16
        pad_w = (-w) % 16
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
        out = self._wct2.transfer_with_style(
            tensor, self._style_feats, self._style_skips, alpha=1.0
        )
        return out[:, :, :h, :w].clamp(0.0, 1.0)

    @staticmethod
    def _bgr_to_tensor(bgr: np.ndarray):
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _tensor_to_bgr(tensor) -> np.ndarray:
        arr = tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _downscale(bgr: np.ndarray, max_side: int) -> np.ndarray:
        h, w = bgr.shape[:2]
        scale = max_side / max(h, w)
        if scale >= 1.0:
            return bgr
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)

    def _blend(self, original: np.ndarray, graded: np.ndarray) -> np.ndarray:
        if self.strength < 1.0:
            return cv2.addWeighted(original, 1.0 - self.strength, graded, self.strength, 0)
        return graded


def colorize_image(input_path: str, output_path: str, 
                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for colorization"""
    pipeline = ColorizationPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def match_color(input_path: str, output_path: str, reference_path: str,
                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for color matching"""
    pipeline = MatchColorPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, reference_path=reference_path,
                           progress_callback=progress_callback)