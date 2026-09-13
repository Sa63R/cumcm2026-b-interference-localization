# Q3/Q4 time-bounded supplemental practice only; frozen formal-test entries; Windows guest desktop, no Mac input events.
$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
$batch=Join-Path $root 'practice_supplement_20260913_1515'
New-Item -ItemType Directory -Force $batch,(Join-Path $batch 'screenshots'),(Join-Path $batch 'runs'),(Join-Path $batch 'official_logs') | Out-Null
$deadline=[DateTimeOffset]::Parse('2026-09-13T09:29:30Z');$caseCap=2000
$pageOffset=44 # Official notice grew by 44 px; verified guest screenshots at 16:14-16:15.
$phase='initializing';$index=0;$completed=@();$question=0
function Write-Json($obj,$path){
 $tmp=$path+'.tmp';$obj | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $tmp
 # Guest-agent reads can briefly hold a sharing lock. Retry replacing only this
 # already-written metadata file; this never retries simulator actions.
 for($attempt=0;$attempt -lt 100;$attempt++){
  try {if(Test-Path $path){[IO.File]::Replace($tmp,$path,[NullString]::Value)}else{[IO.File]::Move($tmp,$path)};return}
  catch {if(-not (Test-Path $tmp)){throw};if($attempt -eq 99){throw};Start-Sleep -Milliseconds 50}
 }
}
function Status($state,$message=''){Write-Json @{status=$state;phase=$phase;index=$index;completed=$completed.Count;target=$caseCap;start_deadline_utc=$deadline.ToString('o');q3_completed=@($completed | Where-Object {$_.question -eq 3}).Count;q4_completed=@($completed | Where-Object {$_.question -eq 4}).Count;question=$question;message=$message;pid=$PID;utc=(Get-Date).ToUniversalTime().ToString('o');last_results=@($completed | Select-Object -Last 3)} (Join-Path $batch 'status.json')}
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
  Start-Sleep -Milliseconds 60
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
  for($listAttempt=0;$listAttempt -lt 30;$listAttempt++){
   Start-Sleep -Milliseconds 250
   $cropY=$(if($question -eq 3){299}else{569})+$pageOffset
   $listText=View $tag 365 $cropY 1185 68
   $expected='开始问题'+$question+'演练'
   if($listText -match $expected -and $listText -match '可以开始' -and $listText -match '干扰源'){return $listText}
  }
  throw "Practice list guard timed out for Q${question}: $listText"
 }
 function Click($x,$y){
  Check-Window
  # Only reviewed practice content controls; sidebar/formal controls excluded.
  if($x -lt 600 -or $x -gt 1550 -or $y -lt 275 -or $y -gt 650){throw 'Click outside reviewed Q4 control area'}
  $screen=[System.Windows.Forms.SystemInformation]::VirtualScreen
  if(-not [Q4BulkUI]::ClickAt($x,$y,$screen.Left,$screen.Top,$screen.Width,$screen.Height)){throw 'Guest SendInput did not insert all click events'}
  Start-Sleep -Milliseconds 300
 }


 $lock=New-Object System.Threading.Mutex($false,'Local\CodexQ3Q4Practice200')
 if(-not $lock.WaitOne(0)){throw 'Another worker already owns this batch'}
 if(@(Get-Process python -ErrorAction SilentlyContinue).Count){throw 'Python is already running; preserve existing test'}
 $expected=@{'run_q3_origin.py'='093b464e69f44b26cae9979ca72ba0e6f0675187e2771fca5f44c0ad81f1778c';'run_official.py'='8d11b7d1ffe709947fa97414599a408fdb5dbf27eb7dc5ecc04540ce9c5d35ad';'source_manifest.json'='fc4c05227c7f5d8ead157d6fb92551f02b0983a35d9477be544105736d3535c5'}
 foreach($f in $expected.Keys){if((Get-FileHash (Join-Path $root $f) -Algorithm SHA256).Hash.ToLower() -ne $expected[$f]){throw "Frozen entry changed: $f"}}
 $createdUtc=if(Test-Path (Join-Path $batch 'batch_config.json')){(Get-Content -Raw -Encoding UTF8 (Join-Path $batch 'batch_config.json') | ConvertFrom-Json).created_utc}else{(Get-Date).ToUniversalTime().ToString('o')}
 Write-Json @{entries=$expected;ui_page_offset_px=$pageOffset;target_per_question=$null;case_cap=$caseCap;start_deadline_utc=$deadline.ToString('o');user_deadline_beijing='2026-09-13 17:30:00';order='alternating Q3 then Q4';mode='official_practice';created_utc=$createdUtc;resumed_utc=(Get-Date).ToUniversalTime().ToString('o')} (Join-Path $batch 'batch_config.json')
 $ledger=Join-Path $batch 'completed.json'
 if(Test-Path $ledger){$completed=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($ledger));if($completed -isnot [Array]){$completed=@($completed)}}
 $done=@{};foreach($row in $completed){$done[[int]$row.index]=$true}
 $logRoot='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs'
 for($index=1;$index -le $caseCap;$index++){
  $question=if($index % 2 -eq 1){3}else{4};$ordinal=[int][math]::Ceiling($index/2.0)
  if($done.ContainsKey($index)){continue}
  if([DateTimeOffset]::UtcNow -ge $deadline){$phase='deadline_reached';Status 'completed' 'Stopped opening new practice cases 30 seconds before user deadline';break}
  if(Test-Path (Join-Path $batch 'STOP')){$phase='between_cases';Status 'stopped';break}
  $prefix=('{0:D3}_q{1}_{2:D3}' -f $index,$question,$ordinal);$runDir=Join-Path $batch ('runs\'+$prefix)
  $assignmentPath=Join-Path $batch ($prefix+'_assignment.json')
  $resumeResult=Test-Path $runDir
  $logFilter='practice-p'+$question+'-*.jlog'
  $before=@{}
  if(-not $resumeResult){
   if(Test-Path $assignmentPath){throw 'Unfinished assigned case exists; review before creating another case'}
   $phase='checking_list';Status 'running'
   $resumeReady=Test-Path (Join-Path $batch ('RESUME_READY_'+$prefix))
   if(-not $resumeReady){$null=Wait-PracticeList ($prefix+'_list')}
   Get-ChildItem $logRoot -Filter $logFilter | ForEach-Object {$before[$_.FullName]=$true}
   $phase='starting_practice';Status 'running'
   if(-not $resumeReady){$startY=$(if($question -eq 3){331}else{601})+$pageOffset;Click 1433 $startY}
   $ready=$false;$label=''
   for($attempt=0;$attempt -lt 90;$attempt++){
    Start-Sleep -Milliseconds 650
    $top=View ($prefix+'_ready') 360 (210+$pageOffset) 1200 200
    $match=[regex]::Match($top,'[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}')
    if($top -match ('问题'+$question+'演练') -and $top -match '尚未进入' -and $top -match '机器狗进入' -and $top -notmatch 'XXXX|倒计时|正在准备|等待测试状态|等存测试状态' -and $top -match '[1-9][0-9][：:]'){$ready=$true;$label=if($match.Success){$match.Value}else{$prefix+'-ui-verified'};break}
   }
   if(-not $ready){throw "Q${question} ready guard timeout: $top"}
   Write-Json @{index=$index;question=$question;ordinal=$ordinal;case_label_ocr=$label;mode='official_practice';text=$top;prior_logs=@($before.Keys);utc=(Get-Date).ToUniversalTime().ToString('o')} $assignmentPath
   $phase='running_policy';Status 'running'
   $entry=if($question -eq 3){'run_q3_origin.py'}else{'run_official.py'}
   $cmd='C:\Users\baiwc\Downloads\Q4Practice\python\python.exe '+$root+'\'+$entry+' --robot-id 202627001104 --case-label '+$label+' --output '+$runDir+' > '+$batch+'\'+$prefix+'_console.txt 2>&1'
   & C:\Windows\System32\cmd.exe /c $cmd
   if($LASTEXITCODE -ne 0){throw 'Policy process failed; preserve case and all evidence'}
  } else {
   $assignment=Get-Content -Raw -Encoding UTF8 $assignmentPath | ConvertFrom-Json
   foreach($p in $assignment.prior_logs){$before[$p]=$true}
   $label=$assignment.case_label_ocr
  }
  $result=Get-Content -Raw -Encoding UTF8 (Join-Path $runDir 'result.json') | ConvertFrom-Json
  if($result.status -ne 'policy_completed_and_exited' -or -not $result.exit_response.accepted -or $null -ne $result.pending_request){throw 'Incomplete strategy outcome'}
  if($question -eq 3 -and -not $result.policy.complete_channel_certificate){throw 'Q3 certificate incomplete'}
  if($question -eq 4 -and $result.policy.stop_certificate -notin @('coverage_complete','source_upper_bound')){throw 'Q4 certificate incomplete'}
  $phase='verifying_official_result';Status 'running';$verified=$false
  for($attempt=0;$attempt -lt 20;$attempt++){
   $dialog=View ($prefix+'_completion') 610 345 500 225
   if($dialog -match ('问题'+$question+'演练.{0,3}完成') -and $dialog -match '/exit' -and $dialog -match '正常结束' -and $dialog -match '日志已保存'){$verified=$true;break}
   Start-Sleep -Milliseconds 300
  }
  if(-not $verified){throw "Completion guard failed: $dialog"}
  $logs=@(Get-ChildItem $logRoot -Filter $logFilter | Where-Object {-not $before.ContainsKey($_.FullName)})
  if($logs.Count -ne 1 -or $logs[0].Length -lt 1){throw 'Expected exactly one new official practice jlog'}
  $sidecar=[IO.Path]::ChangeExtension($logs[0].FullName,'.result.json')
  $official=Get-Content -Raw -Encoding UTF8 $sidecar | ConvertFrom-Json
  $delta=[DateTimeOffset]::Parse($official.ended_at_utc).ToUnixTimeMilliseconds() - [long]$result.exit_response.real_timestamp_ms
  if($official.problem_no -ne $question -or [math]::Abs($delta) -gt 10){throw 'Official metadata does not match accepted exit'}
  $total=[int]$official.jammer_count;$omniCount=[int]$official.omnidirectional_jammer_count;$directionalCount=[int]$official.directional_jammer_count
  if($total -ne $omniCount+$directionalCount -or $total -lt 10 -or $total -gt 16){throw 'Invalid official source counts'}
  if($question -eq 3 -and $directionalCount -ne 0){throw 'Q3 has unexpected directional sources'}
  $totalMatch=[regex]::Match($dialog,'干扰源(\d+)个')
  $uiTotal=if($totalMatch.Success){[int]$totalMatch.Groups[1].Value}else{-1}
  $uiOmni=[regex]::Match($dialog,'(?:全|金)向(\d+)个');$uiDirectional=[regex]::Match($dialog,'定向(\d+)个')
  $uiPartsTotal=if($uiOmni.Success -and $uiDirectional.Success){[int]$uiOmni.Groups[1].Value+[int]$uiDirectional.Groups[1].Value}else{-1}
  if($uiTotal -ne $total -and $uiPartsTotal -ne $total){throw 'Neither OCR total nor sum of source types corroborates official metadata'}
  if($total -ne [int]$result.client_state.cleared_count){throw 'Official count differs from successful clears'}
  $case=[regex]::Match($logs[0].Name,'([A-Z0-9]{4}(?:-[A-Z0-9]{4}){3})\.jlog$').Groups[1].Value
  if(-not $case -or $case -ne $official.case_code){throw 'Missing or conflicting official case code'}
  $hash=(Get-FileHash $logs[0].FullName -Algorithm SHA256).Hash.ToLower()
  if($hash -ne $official.package_sha256){throw 'Official jlog hash mismatch'}
  foreach($ext in @('.jlog','.result.json','.psum')){
   $f=[IO.Path]::ChangeExtension($logs[0].FullName,$ext)
   if(Test-Path $f){Copy-Item $f (Join-Path $batch ('official_logs\'+[IO.Path]::GetFileName($f)))}
  }
  Write-Json @{text=$dialog;total=$total;ui_total=$uiTotal;ui_parts_total=$uiPartsTotal;omni=$omniCount;directional=$directionalCount;official_case=$case;case_label_ocr=$label;exit_timestamp_delta_ms=$delta} (Join-Path $runDir 'ui_verification.json')
  $failed=0;foreach($source in $result.client_state.sources.PSObject.Properties){$failed+=[int]$source.Value.failed_clear_count}
  $cpu=if($null -ne $result.cpu_seconds){[double]$result.cpu_seconds}else{$null}
  $row=@{index=$index;question=$question;ordinal=$ordinal;method=$(if($question -eq 3){'v3_origin20'}else{'v6_lite'});case=$case;case_label_ocr=$label;total=$total;omni=$omniCount;directional=$directionalCount;cleared=[int]$result.client_state.cleared_count;virtual_s=[double]$result.client_state.virtual_time_s;seconds_per_source=[double]$result.seconds_per_accepted_clear;wall_s=[double]$result.wall_seconds;cpu_s=$cpu;actions=[int]$result.client_state.accepted_actions;failed_clear_attempts=$failed;official_log=$logs[0].Name;official_log_sha256=$hash;utc=(Get-Date).ToUniversalTime().ToString('o')}
  Write-Json $row (Join-Path $runDir 'completed_record.json')
  $completed+=,$row;Write-Json @($completed) $ledger
  $phase='returning_to_list';Status 'running';Click 1018 531
  $returned=$false
  for($attempt=0;$attempt -lt 15;$attempt++){
   $page=View ($prefix+'_done_page') 360 210 1200 680
   if($page -match ('问题'+$question+'演练') -and $page -match '返回演练' -and $page -match '行为日志已保存'){$returned=$true;break}
   Start-Sleep -Milliseconds 250
  }
  if(-not $returned){throw 'Return page guard failed'}
  Click 1466 (330+$pageOffset)
  $null=Wait-PracticeList ($prefix+'_return_list')
 }
 if($completed.Count -eq $caseCap){$phase='finished';Status 'completed'}
} catch {Status 'error' ($_.Exception.ToString()+' '+$_.InvocationInfo.PositionMessage)}
finally {
 if($lock){try{$lock.ReleaseMutex()}catch{};$lock.Dispose()}
 [Q4BulkUI]::SetThreadExecutionState(2147483648) | Out-Null
}
