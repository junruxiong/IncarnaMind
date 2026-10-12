# The Windows system OCR for the OCR comparison (eval/ocr/README.md):
# Windows.Media.Ocr, as an app calls it. It reads one language at a time:
# the first of the User's profile languages that has an OCR pack installed
# (Settings > Time & language > Language), or the one given, if its pack is.
# Prints the page's lines, top to bottom, then "## <milliseconds>" the
# recognition took.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File eval/ocr/windows-ocr.ps1 page.jpg [zh-Hans-CN]
param(
  [Parameter(Mandatory = $true)][string]$Path,
  [string]$Language = ""
)

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]

$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
  })[0]
function Await($operation, [Type]$type) {
  $task = $asTask.MakeGenericMethod($type).Invoke($null, @($operation))
  $task.Wait(-1) | Out-Null
  $task.Result
}

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync((Resolve-Path $Path).Path)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

$engine = if ($Language) {
  [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Language))
} else {
  [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}
if ($null -eq $engine) { throw "No OCR pack is installed for $(if ($Language) { $Language } else { 'the user''s languages' })." }
$watch = [Diagnostics.Stopwatch]::StartNew()
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$watch.Stop()
$result.Lines | ForEach-Object { $_.Text }
"## $($watch.ElapsedMilliseconds)"
