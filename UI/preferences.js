(() => {
    const applySettings = settings => {
        const root = document.documentElement;
        const theme = settings.theme || 'dark';
        const applyTheme = () => {
            const resolved = theme === 'system'
                ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
                : theme;
            root.setAttribute('data-theme', resolved);
        };

        applyTheme();
        const accentColor = settings.accentColor || '#3b82f6';
        root.style.setProperty('--primary', accentColor);
        root.style.setProperty('--primary-hover', darkenColor(accentColor, 0.15));
        const fontSizes = { small: '12px', medium: '14px', large: '16px' };
        root.style.setProperty('--base-font-size', fontSizes[settings.fontSize] || fontSizes.medium);
        root.setAttribute('data-density', settings.density || 'comfortable');

        if (theme === 'system') {
            const media = window.matchMedia('(prefers-color-scheme: dark)');
            media.addEventListener('change', applyTheme);
        }
    };

    const darkenColor = (hex, amount) => {
        const value = Number.parseInt(hex.replace('#', ''), 16);
        const channel = shift => Math.max(0, Math.floor(((value >> shift) & 0xff) * (1 - amount)));
        return `#${[channel(16), channel(8), channel(0)].map(c => c.toString(16).padStart(2, '0')).join('')}`;
    };

    document.addEventListener('DOMContentLoaded', async () => {
        try {
            const response = await fetch('/api/settings');
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            applySettings(await response.json());
        } catch (error) {
            console.error('Could not load application appearance settings:', error);
        }
    });
})();
