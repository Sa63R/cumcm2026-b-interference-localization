# Runs only inside the logged-in WINDOWS GUEST session. No host input events.
$ErrorActionPreference='Stop'
trap { @{status='error';error=$_.Exception.ToString()} | ConvertTo-Json | Set-Content -Encoding UTF8 'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\desktop_server_status.json'; exit 1 }
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$requests=Join-Path $root 'desktop_requests'
$results=Join-Path $root 'desktop_results'
New-Item -ItemType Directory -Force $requests,$results | Out-Null
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class Q4Desktop {
 [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left,Top,Right,Bottom; }
 [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr value);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hwnd,int cmd);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd,out RECT rect);
 [StructLayout(LayoutKind.Sequential)] public struct MOUSEINPUT {public int dx,dy;public uint mouseData,flags,time;public UIntPtr extra;}
 [StructLayout(LayoutKind.Sequential)] public struct INPUT {public uint type;public MOUSEINPUT mouse;}
 [DllImport("user32.dll",SetLastError=true)] public static extern uint SendInput(uint count,INPUT[] inputs,int size);
 public static bool ClickAt(int x,int y,int left,int top,int width,int height){
 var inputs=new INPUT[3];
 inputs[0].mouse.dx=(int)Math.Round((x-left)*65535.0/(width-1));
 inputs[0].mouse.dy=(int)Math.Round((y-top)*65535.0/(height-1));
 inputs[0].mouse.flags=0xC001;inputs[1].mouse.flags=2;inputs[2].mouse.flags=4;
 return SendInput(3,inputs,Marshal.SizeOf(typeof(INPUT)))==3;
 }
}
'@
[Q4Desktop]::SetProcessDpiAwarenessContext([IntPtr](-4)) | Out-Null
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
$deadline=(Get-Date).AddMinutes(30)
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
   if($request.action -eq 'maximize'){[Q4Desktop]::ShowWindow($hwnd,3) | Out-Null}
   [Q4Desktop]::SetForegroundWindow($hwnd) | Out-Null
   Start-Sleep -Milliseconds 150
   if([Q4Desktop]::GetForegroundWindow() -ne $hwnd){throw 'Official simulator is not the foreground guest window'}
   $rect=New-Object Q4Desktop+RECT
   [Q4Desktop]::GetWindowRect($hwnd,[ref]$rect) | Out-Null
   switch($request.action) {
    'click' {
      $x=[int]$request.x;$y=[int]$request.y
      if($x -lt $rect.Left -or $x -ge $rect.Right -or $y -lt $rect.Top -or $y -ge $rect.Bottom){throw 'Requested point is outside the simulator window'}
      $screen=[System.Windows.Forms.SystemInformation]::VirtualScreen
      if(-not [Q4Desktop]::ClickAt($x,$y,$screen.Left,$screen.Top,$screen.Width,$screen.Height)){throw 'Guest click failed'}
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
