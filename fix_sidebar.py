with open('UI/src/components/PipelineSidebar.tsx', 'r') as f:
    content = f.read()

# Fix the return opening
content = content.replace(
    '  return (\n    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg-panel)]">',
    '  return (\n    <>\n      <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg-panel)]">'
)

# Fix the closing
old_close = '''    </div>

    <ExportDialog
      isOpen={exportDialogOpen}
      onClose={() => setExportDialogOpen(false)}
      onStartExport={handleStartExport}
      selectedFile={selectedFiles?.[0] || null}
      steps={steps}
    />
  )'''
new_close = '''    </div>

    <ExportDialog
      isOpen={exportDialogOpen}
      onClose={() => setExportDialogOpen(false)}
      onStartExport={handleStartExport}
      selectedFile={selectedFiles?.[0] || null}
      steps={steps}
    />
  </>
  )'''
content = content.replace(old_close, new_close)

with open('UI/src/components/PipelineSidebar.tsx', 'w') as f:
    f.write(content)

print('Done')