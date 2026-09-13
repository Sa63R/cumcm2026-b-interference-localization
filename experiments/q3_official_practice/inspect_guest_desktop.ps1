# Read-only guest-session screenshot and foreground-window diagnostics.
$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q3Practice'
Add-Type -AssemblyName System.Drawing,System.Windows.Forms
Add-Type @'
using System;using System.Runtime.InteropServices;using System.Text;
public class Q3Inspect {
 [DllImport("user32.dll")]public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")]public static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
 [DllImport("user32.dll")]public static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);
 [DllImport("user32.dll")]public static extern bool SetProcessDpiAwarenessContext(IntPtr h);
}
'@
[Q3Inspect]::SetProcessDpiAwarenessContext([IntPtr](-4)) | Out-Null
$hwnd=[Q3Inspect]::GetForegroundWindow();$title=New-Object System.Text.StringBuilder(512);[uint32]$owner=0
[Q3Inspect]::GetWindowText($hwnd,$title,512)|Out-Null
[Q3Inspect]::GetWindowThreadProcessId($hwnd,[ref]$owner)|Out-Null
$bounds=[System.Windows.Forms.SystemInformation]::VirtualScreen
$bitmap=New-Object System.Drawing.Bitmap($bounds.Width,$bounds.Height)
$graphics=[System.Drawing.Graphics]::FromImage($bitmap)
$path=Join-Path $root 'guest_inspection.png'
try{$graphics.CopyFromScreen($bounds.Left,$bounds.Top,0,0,$bounds.Size);$bitmap.Save($path,[System.Drawing.Imaging.ImageFormat]::Png)}finally{$graphics.Dispose();$bitmap.Dispose()}
@{time=(Get-Date).ToString('o');foreground_title=$title.ToString();foreground_pid=$owner;foreground_handle=$hwnd.ToInt64();foreground_process=(Get-Process -Id $owner -ErrorAction SilentlyContinue).ProcessName;session=(Get-Process -Id $PID).SessionId;image=$path} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'guest_inspection.json')
