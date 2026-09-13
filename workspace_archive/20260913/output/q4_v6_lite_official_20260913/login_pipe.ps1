$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
try {
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

 $pipe=[IO.Pipes.NamedPipeServerStream]::new('CodexQ4Login-20260913-904b',[IO.Pipes.PipeDirection]::In,1,[IO.Pipes.PipeTransmissionMode]::Byte,[IO.Pipes.PipeOptions]::Asynchronous)
 $wait=$pipe.BeginWaitForConnection($null,$null)
 if(-not $wait.AsyncWaitHandle.WaitOne(90000)){throw 'Login input timed out'}
 $pipe.EndWaitForConnection($wait)
 $reader=[IO.StreamReader]::new($pipe)
 $login=($reader.ReadLine() | ConvertFrom-Json)
 $reader.Dispose();$pipe.Dispose()
 $sim=Get-Process -Name jammers-simulator
 if($sim.Count -ne 1 -or $sim.Path -ne 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe'){throw 'Unexpected program'}
 [Q4Desktop]::SetForegroundWindow($sim.MainWindowHandle)|Out-Null
 Start-Sleep -Milliseconds 150
 if([Q4Desktop]::GetForegroundWindow() -ne $sim.MainWindowHandle){throw 'Wrong guest foreground'}
 $bounds=[System.Windows.Forms.SystemInformation]::VirtualScreen
 if($bounds.Width -ne 1710){throw 'Unexpected guest layout'}
 $rect=New-Object Q4Desktop+RECT
 [Q4Desktop]::GetWindowRect($sim.MainWindowHandle,[ref]$rect)|Out-Null
 if($rect.Left -ne -8 -or $rect.Top -ne -8){throw 'Expected maximized simulator'}
 if(-not [Q4Desktop]::ClickAt(805,450,0,0,$bounds.Width,$bounds.Height)){throw 'Guest click failed'}
 [System.Windows.Forms.SendKeys]::SendWait([string]$login.account)
 if(-not [Q4Desktop]::ClickAt(805,525,0,0,$bounds.Width,$bounds.Height)){throw 'Guest click failed'}
 [System.Windows.Forms.SendKeys]::SendWait([string]$login.password)
 $login=$null
 if(-not [Q4Desktop]::ClickAt(850,632,0,0,$bounds.Width,$bounds.Height)){throw 'Guest click failed'}
 @{status='login_submitted';utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Out-File "$root\login_status.json" -Encoding utf8
} catch {@{status='error';message=$_.Exception.Message} | ConvertTo-Json | Out-File "$root\login_status.json" -Encoding utf8}
