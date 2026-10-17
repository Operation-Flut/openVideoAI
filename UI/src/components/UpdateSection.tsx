import { useState } from 'react'
import { Download, RefreshCw, AlertCircle, CheckCircle, ArrowUpRight, Loader2 } from 'lucide-react'
import { Button, Badge } from '../components/ui'
import { useUpdateCheck, usePreReleaseCheck, useInstallUpdate } from '../hooks/useUpdate'
import { clsx } from 'clsx'

export function UpdateSection() {
  const { data: updateCheck, isLoading: checkLoading, refetch: refetchCheck } = useUpdateCheck()
  const { data: preReleaseCheck, isLoading: preLoading, refetch: refetchPre } = usePreReleaseCheck()
  const installUpdate = useInstallUpdate()

  const [installingTag, setInstallingTag] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<{ loaded: number; total: number } | null>(null)

  const hasStableUpdate = updateCheck?.update_available
  const hasPreRelease = preReleaseCheck?.pre_release_available

  const currentVersion = updateCheck?.current_version || preReleaseCheck?.current_version || 'unknown'

  const handleCheckNow = () => {
    refetchCheck()
    refetchPre()
  }

  const handleInstall = async (tag: string) => {
    setInstallingTag(tag)
    setDownloadProgress({ loaded: 0, total: 0 })
    try {
      await installUpdate.mutateAsync(tag)
      // The mutation returns the local path; we could open the folder here
      alert(`Downloaded to ${installUpdate.data?.asset_path}. Run the installer manually.`)
    } catch (error) {
      console.error('Install failed:', error)
      alert(`Download failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setInstallingTag(null)
      setDownloadProgress(null)
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-[13px] font-semibold text-[var(--color-text-primary)]">Updates</h3>
        <Button variant="ghost" size="sm" onClick={handleCheckNow} disabled={checkLoading || preLoading}>
          <RefreshCw className={clsx('w-3.5 h-3.5', checkLoading && 'animate-spin')} />
          <span className="ml-1">Check now</span>
        </Button>
      </div>

      <div className="text-[12px] text-[var(--color-text-muted)] font-mono">
        Current version: <span className="text-[var(--color-text-secondary)]">{currentVersion}</span>
      </div>

      {/* Stable release */}
      {checkLoading ? (
        <div className="flex items-center gap-3 p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-[var(--radius-md)]">
          <Loader2 className="w-4 h-4 animate-spin text-[var(--color-accent)]" />
          <span className="text-[13px] text-[var(--color-text-secondary)]">Checking for stable release…</span>
        </div>
      ) : updateCheck?.error ? (
        <div className="flex items-center gap-2 p-3 bg-[var(--color-error-soft)] border border-[var(--color-error)]/30 rounded-[var(--radius-md)] text-[12px] text-[var(--color-error)]">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>Failed to check for updates: {updateCheck.error}</span>
        </div>
      ) : hasStableUpdate ? (
        <div className="flex flex-col gap-2 p-3 bg-[var(--color-success-soft)] border border-[var(--color-success)]/30 rounded-[var(--radius-md)]">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-[var(--color-success)] flex-shrink-0" />
            <span className="text-[13px] font-medium text-[var(--color-success)]">
              Update available: {updateCheck.latest_version}
            </span>
            <Badge variant="success" size="sm">Stable</Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              onClick={() => handleInstall(updateCheck.latest_version!)}
              disabled={installingTag === updateCheck.latest_version || installUpdate.isPending}
            >
              {installingTag === updateCheck.latest_version && (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                  Downloading…
                </>
              )}
              {installingTag !== updateCheck.latest_version && (
                <>
                  <Download className="w-3.5 h-3.5 mr-1.5" />
                  Download installer
                </>
              )}
            </Button>
            {downloadProgress && (
              <span className="text-[11px] text-[var(--color-text-muted)] font-mono">
                {downloadProgress.loaded} / {downloadProgress.total} bytes
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2 p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-[var(--radius-md)]">
          <CheckCircle className="w-4 h-4 text-[var(--color-success)] flex-shrink-0" />
          <span className="text-[13px] text-[var(--color-text-secondary)]">
            You are on the latest stable release.
          </span>
        </div>
      )}

      {/* Pre-release */}
      {preLoading ? (
        <div className="flex items-center gap-3 p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-[var(--radius-md)]">
          <Loader2 className="w-4 h-4 animate-spin text-[var(--color-accent)]" />
          <span className="text-[13px] text-[var(--color-text-secondary)]">Checking for pre-release…</span>
        </div>
      ) : preReleaseCheck?.error ? (
        <div className="flex items-center gap-2 p-3 bg-[var(--color-error-soft)] border border-[var(--color-error)]/30 rounded-[var(--radius-md)] text-[12px] text-[var(--color-error)]">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>Failed to check for pre-releases: {preReleaseCheck.error}</span>
        </div>
      ) : hasPreRelease ? (
        <div className="flex flex-col gap-2 p-3 bg-[var(--color-warning-soft)] border border-[var(--color-warning)]/30 rounded-[var(--radius-md)]">
          <div className="flex items-center gap-2">
            <ArrowUpRight className="w-4 h-4 text-[var(--color-warning)] flex-shrink-0" />
            <span className="text-[13px] font-medium text-[var(--color-warning)]">
              Pre-release available: {preReleaseCheck.latest_pre_release_version}
            </span>
            <Badge variant="warning" size="sm">Pre-release</Badge>
            {preReleaseCheck.latest_pre_release_date && (
              <span className="text-[11px] text-[var(--color-text-muted)] ml-auto">
                Published {new Date(preReleaseCheck.latest_pre_release_date).toLocaleDateString()}
              </span>
            )}
          </div>
          <div className="text-[12px] text-[var(--color-text-muted)]">
            Pre-releases may contain new features but can be less stable. Install at your own risk.
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => handleInstall(preReleaseCheck.latest_pre_release_version!)}
            disabled={installingTag === preReleaseCheck.latest_pre_release_version || installUpdate.isPending}
          >
            {installingTag === preReleaseCheck.latest_pre_release_version && (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                Downloading…
              </>
            )}
            {installingTag !== preReleaseCheck.latest_pre_release_version && (
              <>
                <Download className="w-3.5 h-3.5 mr-1.5" />
                Download pre-release
              </>
            )}
          </Button>
        </div>
      ) : (
        <div className="flex items-center gap-2 p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-[var(--radius-md)]">
          <CheckCircle className="w-4 h-4 text-[var(--color-success)] flex-shrink-0" />
          <span className="text-[13px] text-[var(--color-text-secondary)]">
            No pre-release newer than your version.
          </span>
        </div>
      )}

      <div className="text-[11px] text-[var(--color-text-muted)] border-t border-[var(--color-border)] pt-3">
        Checks <a href="https://github.com/Operation-Flut/openVideoAI/releases" target="_blank" rel="noopener noreferrer" className="underline hover:text-[var(--color-accent)]">GitHub Releases</a>.
        Stable and pre-release checks run independently; a pre-release is only shown when it is
        newer than your current version.
      </div>
    </section>
  )
}