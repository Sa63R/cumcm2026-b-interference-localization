# Q4 practice only; Windows guest desktop, no Mac input events.
$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$batch=Join-Path $root 'additional_10'
New-Item -ItemType Directory -Force $batch,(Join-Path $batch 'screenshots'),(Join-Path $batch 'runs'),(Join-Path $batch 'official_logs') | Out-Null
$phase='initializing';$index=0;$completed=@()
function Write-Json($obj,$path){$tmp=$path+'.tmp';$obj | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $tmp;Move-Item -Force $tmp $path}
function Status($state,$message=''){Write-Json @{status=$state;phase=$phase;index=$index;completed=$completed.Count;target=10;message=$message;pid=$PID;utc=(Get-Date).ToUniversalTime().ToString('o');last_results=@($completed | Select-Object -Last 3)} (Join-Path $batch 'status.json')}
try {
 Add-Type -AssemblyName System.Drawing,System.Windows.Forms,System.Runtime.WindowsRuntime
 Add-Type @'
using System;using System.Runtime.InteropServices;
public class Q4BulkUI {
 [StructLayout(LayoutKind.Sequential)] public struct RECT {public int Left,Top,Right,Bottom;}
 [StructLayout(LayoutKind.Sequential)] public struct MOUSEINPUT {public int dx,dy;public uint mouseData,flags,time;public UIntPtr extra;}
 [StructLayout(LayoutKind.Sequential)] public struct INPUT {public uint type;public MOUSEINPUT mouse;}
 [DllImport("user32.dll")]public static extern bool SetProcessDpiAwarenessContext(IntPtr value);
 [DllImport("user32.dll")]public static extern bool SetForegroundWindow(IntPtr hwnd);
 [DllImport("user32.dll")]public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")]public static extern bool GetWindowRect(IntPtr hwnd,out RECT rect);
 [DllImport("user32.dll")]public static extern bool SetCursorPos(int x,int y);
 [DllImport("user32.dll")]public static extern void mouse_event(uint flags,uint x,uint y,uint data,UIntPtr extra);
 [DllImport("user32.dll",SetLastError=true)]public static extern uint SendInput(uint count,INPUT[] inputs,int size);
 public static bool ClickAt(int x,int y,int left,int top,int width,int height){
  var inputs=new INPUT[3];
  inputs[0].mouse.dx=(int)Math.Round((x-left)*65535.0/(width-1));
  inputs[0].mouse.dy=(int)Math.Round((y-top)*65535.0/(height-1));
  inputs[0].mouse.flags=0xC001; // MOVE | ABSOLUTE | VIRTUALDESK
  inputs[1].mouse.flags=2;inputs[2].mouse.flags=4;
  return SendInput(3,inputs,Marshal.SizeOf(typeof(INPUT)))==3;
 }
 [DllImport("kernel32.dll")]public static extern uint SetThreadExecutionState(uint flags);
}
'@
 [Q4BulkUI]::SetProcessDpiAwarenessContext([IntPtr](-4)) | Out-Null
 [Q4BulkUI]::SetThreadExecutionState(2147483651) | Out-Null
 $null=[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
 $null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
 $null=[Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
 $null=[Windows.Graphics.Imaging.SoftwareBitmap,Windows.Foundation,ContentType=WindowsRuntime]
 $null=[Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
 $null=[Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
 $asTask=([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {$_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.IsGenericMethod -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'})[0]
 function Await($operation,$type){$task=$asTask.MakeGenericMethod($type).Invoke($null,@($operation));$task.Wait();return $task.Result}
 $engine=[Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
 if(-not $engine){throw 'No Windows OCR recognizer available'}
 function Check-Window {
  $sim=Get-Process -Name jammers-simulator | Where-Object {$_.SessionId -eq (Get-Process -Id $PID).SessionId} | Select-Object -First 1
  if(-not $sim -or $sim.Path -ne 'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe'){throw 'Expected official simulator not found in guest session'}
  $script:hwnd=$sim.MainWindowHandle
  [Q4BulkUI]::SetForegroundWindow($hwnd) | Out-Null
  Start-Sleep -Milliseconds 100
  if([Q4BulkUI]::GetForegroundWindow() -ne $hwnd){throw 'Guest simulator not foreground'}
  $rect=New-Object Q4BulkUI+RECT
  [Q4BulkUI]::GetWindowRect($hwnd,[ref]$rect) | Out-Null
  $screen=[System.Windows.Forms.SystemInformation]::VirtualScreen
  if($rect.Left -ne -8 -or $rect.Top -ne -8 -or $screen.Width -ne 1710 -or $screen.Height -lt 880){throw 'Guest layout changed; stop for review'}
 }
 function View($tag,$x,$y,$w,$h){
  Check-Window
  $path=Join-Path $batch ('screenshots\'+$tag+'.png')
  $raw=New-Object System.Drawing.Bitmap($w,$h)
  $g=[System.Drawing.Graphics]::FromImage($raw)
  try{$g.CopyFromScreen($x,$y,0,0,$raw.Size)}finally{$g.Dispose()}
  $big=New-Object System.Drawing.Bitmap(($w*2),($h*2))
  $g=[System.Drawing.Graphics]::FromImage($big)
  try{$g.DrawImage($raw,0,0,$big.Width,$big.Height);$big.Save($path,[System.Drawing.Imaging.ImageFormat]::Png)}finally{$g.Dispose();$big.Dispose();$raw.Dispose()}
  $file=Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
  $stream=Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
  try{
   $decoder=Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
   $bitmap=Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
   try{$ocr=Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])}finally{$bitmap.Dispose()}
   $text=($ocr.Lines.Text -join '|') -replace '\s',''
   $text=$text.Replace('測','测').Replace('試','试').Replace('練','练').Replace('問','问').Replace('題','题').Replace('誌','志').Replace('數','数')
   Write-Json @{text=$text;image=$path;x=$x;y=$y;width=$w;height=$h} ($path+'.json')
   return $text
  }finally{$stream.Dispose()}
 }
 function Wait-PracticeList($tag){
  for($listAttempt=0;$listAttempt -lt 20;$listAttempt++){
   Start-Sleep -Milliseconds 500
   $listText=View $tag 365 545 1185 68
   if($listText -match '开始问题4演练' -and $listText -match '包含定向干扰源'){return $listText}
  }
  throw "Q4 practice list guard timed out: $listText"
 }
 function Click($x,$y){
  Check-Window
  # All permitted controls are in the Q4 content area; sidebar/formal controls excluded.
  if($x -lt 600 -or $x -gt 1550 -or $y -lt 275 -or $y -gt 650){throw 'Click outside reviewed Q4 control area'}
  $screen=[System.Windows.Forms.SystemInformation]::VirtualScreen
  if(-not [Q4BulkUI]::ClickAt($x,$y,$screen.Left,$screen.Top,$screen.Width,$screen.Height)){throw 'Guest SendInput did not insert all click events'}
  Start-Sleep -Milliseconds 700
 }

 $ledger=Join-Path $batch 'completed.json'
 if(Test-Path $ledger){$completed=@(Get-Content -Raw -Encoding UTF8 $ledger | ConvertFrom-Json)}
 $done=@{};foreach($row in $completed){$done[[int]$row.index]=$true}
 $logRoot='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs'
 for($index=1;$index -le 10;$index++){
  if($done.ContainsKey($index)){continue}
  if(Test-Path (Join-Path $batch 'STOP')){$phase='between_cases';Status 'stopped';break}
  $prefix=('{0:D2}' -f $index);$runDir=Join-Path $batch ('runs\'+$prefix)
  if(Test-Path $runDir){throw 'Existing uncommitted result requires review'}
  $phase='checking_list';Status 'running'
  $null=Wait-PracticeList ($prefix+'_list')
  $before=@{};Get-ChildItem $logRoot -Filter 'practice-p4-*.jlog' | ForEach-Object {$before[$_.FullName]=$true}
  $phase='starting_practice';Status 'running';Click 1433 578
  $ready=$false;$label=''
  for($attempt=0;$attempt -lt 50;$attempt++){
   Start-Sleep -Seconds 2
   $top=View ($prefix+'_ready') 360 210 1200 145
   $match=[regex]::Match($top,'[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}')
   if($top -match '问题4演练' -and $top -match '尚未进入' -and $top -match '等待机器狗进入' -and $match.Success -and $match.Value -notmatch 'XXXX' -and $top -notmatch '倒计时|正在准备'){$ready=$true;$label=$match.Value;break}
  }
  if(-not $ready){throw "Q4 ready guard timeout: $top"}
  Write-Json @{case_label_ocr=$label;mode='official_q4_practice';text=$top;utc=(Get-Date).ToUniversalTime().ToString('o')} (Join-Path $batch ($prefix+'_assignment.json'))
  $phase='running_v6_lite';Status 'running'
  $cmd='C:\Users\baiwc\Downloads\Q4Practice\python\python.exe '+$root+'\run_official.py --robot-id 202627001104 --case-label '+$label+' --practice-confirmed --output '+$runDir+' > '+$batch+'\'+$prefix+'_console.txt 2>&1'
  & C:\Windows\System32\cmd.exe /c $cmd
  if($LASTEXITCODE -ne 0){throw 'Policy process failed; preserve current case'}
  $result=Get-Content -Raw -Encoding UTF8 (Join-Path $runDir 'result.json') | ConvertFrom-Json
  if($result.status -ne 'policy_completed_and_exited' -or -not $result.exit_response.accepted -or $null -ne $result.pending_request){throw 'Incomplete strategy outcome'}
  $phase='verifying_official_result';Status 'running';$verified=$false
  for($attempt=0;$attempt -lt 20;$attempt++){
   $dialog=View ($prefix+'_completion') 610 345 500 225
   if($dialog -match '问题4演练.{0,3}完成' -and $dialog -match '/exit' -and $dialog -match '正常结束' -and $dialog -match '日志已保存'){$verified=$true;break}
   Start-Sleep -Milliseconds 500
  }
  if(-not $verified){throw "Completion guard failed: $dialog"}
  $omni=[regex]::Match($dialog,'全向(\d+)个');$directional=[regex]::Match($dialog,'定向(\d+)个')
  $totalMatch=[regex]::Match($dialog,'干扰源(\d+)个')
  if(-not $omni.Success -or -not $directional.Success){throw "Source types unreadable: $dialog"}
  $total=[int]$omni.Groups[1].Value+[int]$directional.Groups[1].Value
  if($totalMatch.Success -and [int]$totalMatch.Groups[1].Value -ne $total){throw 'Conflicting official source counts'}
  if($total -ne [int]$result.client_state.cleared_count){throw 'Official count differs from successful clears'}
  $logs=@(Get-ChildItem $logRoot -Filter 'practice-p4-*.jlog' | Where-Object {-not $before.ContainsKey($_.FullName)})
  if($logs.Count -ne 1 -or $logs[0].Length -lt 1){throw 'Expected exactly one new official Q4 jlog'}
  $case=[regex]::Match($logs[0].Name,'([A-Z0-9]{4}(?:-[A-Z0-9]{4}){3})\.jlog$').Groups[1].Value
  if(-not $case){throw 'Missing official case code'}
  Copy-Item $logs[0].FullName (Join-Path $batch ('official_logs\'+$logs[0].Name))
  $sidecar=[IO.Path]::ChangeExtension($logs[0].FullName,'.result.json')
  if(Test-Path $sidecar){Copy-Item $sidecar (Join-Path $batch ('official_logs\'+[IO.Path]::GetFileName($sidecar)))}
  Write-Json @{text=$dialog;total=$total;omni=[int]$omni.Groups[1].Value;directional=[int]$directional.Groups[1].Value;official_case=$case;case_label_ocr=$label} (Join-Path $runDir 'ui_verification.json')
  $failed=0;foreach($source in $result.client_state.sources.PSObject.Properties){$failed+=[int]$source.Value.failed_clear_count}
  $row=@{index=$index;case=$case;case_label_ocr=$label;total=$total;omni=[int]$omni.Groups[1].Value;directional=[int]$directional.Groups[1].Value;cleared=[int]$result.client_state.cleared_count;virtual_s=[double]$result.client_state.virtual_time_s;seconds_per_source=[double]$result.seconds_per_accepted_clear;wall_s=[double]$result.wall_seconds;cpu_s=[double]$result.cpu_seconds;actions=[int]$result.client_state.accepted_actions;failed_clear_attempts=$failed;official_log=$logs[0].Name;official_log_sha256=(Get-FileHash $logs[0].FullName -Algorithm SHA256).Hash.ToLower();utc=(Get-Date).ToUniversalTime().ToString('o')}
  $completed+=,$row;Write-Json @($completed) $ledger
  $phase='returning_to_list';Status 'running';Click 1018 531
  $page=View ($prefix+'_done_page') 360 210 1200 680
  if($page -notmatch '问题4演练' -or $page -notmatch '返回演练' -or $page -notmatch '行为日志已保存'){throw 'Return guard failed'}
  Click 1466 308
  $null=Wait-PracticeList ($prefix+'_return_list')
 }
 if($completed.Count -eq 10){$phase='finished';Status 'completed'}
} catch {Status 'error' ($_.Exception.ToString()+' '+$_.InvocationInfo.PositionMessage)}
finally {[Q4BulkUI]::SetThreadExecutionState(2147483648) | Out-Null}
