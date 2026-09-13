# Runs only inside the logged-in WINDOWS GUEST session. No host input events.
$ErrorActionPreference='Stop'
trap { @{status='error';error=$_.Exception.ToString()} | ConvertTo-Json | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\Q3Practice\desktop_server_status.json'; exit 1 }
$root='C:\Users\baiwc\Downloads\Q3Practice'
$requests=Join-Path $root 'desktop_requests'
$results=Join-Path $root 'desktop_results'
New-Item -ItemType Directory -Force $requests,$results | Out-Null
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class Q3Desktop {
 [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left,Top,Right,Bottom; }
 [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr value);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hwnd,int cmd);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd,out RECT rect);
 [DllImport("user32.dll")] public static extern bool SetCursorPos(int x,int y);
 [DllImport("user32.dll")] public static extern void mouse_event(uint flags,uint x,uint y,uint data,UIntPtr extra);
 [DllImport("user32.dll")] public static extern void keybd_event(byte key,byte scan,uint flags,UIntPtr extra);
}
'@
[Q3Desktop]::SetProcessDpiAwarenessContext([IntPtr](-4)) | Out-Null
function Save-Screenshot($path) {
 $bounds=[System.Windows.Forms.SystemInformation]::VirtualScreen
 $bitmap=New-Object System.Drawing.Bitmap($bounds.Width,$bounds.Height)
 $graphics=[System.Drawing.Graphics]::FromImage($bitmap)
 try {
  $graphics.CopyFromScreen($bounds.Left,$bounds.Top,0,0,$bounds.Size)
  $bitmap.Save($path,[System.Drawing.Imaging.ImageFormat]::Png)
 } finally {$graphics.Dispose();$bitmap.Dispose()}
 return @{left=$bounds.Left;top=$bounds.Top;width=$bounds.Width;height=$bounds.Height}
}
@{status='ready';session=(Get-Process -Id $PID).SessionId;pid=$PID} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'desktop_server_status.json')
$deadline=(Get-Date).AddMinutes(90)
$quit=$false
while ((Get-Date) -lt $deadline -and -not $quit) {
 foreach($file in @(Get-ChildItem $requests -Filter '*.json' | Sort-Object Name)) {
  $response=Join-Path $results $file.Name
  if(Test-Path $response){continue}
  try {
   $request=Get-Content -Raw $file.FullName | ConvertFrom-Json
   $sim=Get-Process -Name jammers-simulator | Where-Object {$_.SessionId -eq (Get-Process -Id $PID).SessionId} | Select-Object -First 1
   if(-not $sim -or $sim.Path -ne 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe'){throw 'Expected official simulator process not found'}
   $hwnd=$sim.MainWindowHandle
   if($hwnd -eq 0){throw 'Simulator has no desktop window'}
   if($request.action -eq 'maximize'){[Q3Desktop]::ShowWindow($hwnd,3) | Out-Null}
   [Q3Desktop]::SetForegroundWindow($hwnd) | Out-Null
   Start-Sleep -Milliseconds 150
   if([Q3Desktop]::GetForegroundWindow() -ne $hwnd){throw 'Official simulator is not the foreground guest window'}
   $rect=New-Object Q3Desktop+RECT
   [Q3Desktop]::GetWindowRect($hwnd,[ref]$rect) | Out-Null
   switch($request.action) {
    'click' {
      $x=[int]$request.x;$y=[int]$request.y
      if($x -lt $rect.Left -or $x -ge $rect.Right -or $y -lt $rect.Top -or $y -ge $rect.Bottom){throw 'Requested point is outside the simulator window'}
      [Q3Desktop]::SetCursorPos($x,$y) | Out-Null
      Start-Sleep -Milliseconds 100
      [Q3Desktop]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
      Start-Sleep -Milliseconds 70
      [Q3Desktop]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
    }
    'snapshot' {}
    'maximize' {}
    'quit' {$quit=$true}
    default {throw 'Unsupported action'}
   }
   # Wait only inside the guest worker; the Mac remains free for normal use.
   Start-Sleep -Milliseconds 1300
   $png=Join-Path $results ($file.BaseName+'.png')
   $bounds=Save-Screenshot $png
   @{status='ok';id=$request.id;action=$request.action;screen=$bounds;window=@{left=$rect.Left;top=$rect.Top;right=$rect.Right;bottom=$rect.Bottom};image=$png;recorded_at=(Get-Date).ToString('o')} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $response
  } catch { @{status='error';error=$_.Exception.ToString();id=$file.BaseName} | ConvertTo-Json | Set-Content -Encoding UTF8 $response }
 }
 Start-Sleep -Milliseconds 200
}
@{status='stopped';session=(Get-Process -Id $PID).SessionId;pid=$PID} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'desktop_server_status.json')
