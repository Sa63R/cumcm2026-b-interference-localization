# Q3 PRACTICE only. Runs entirely in the logged-in Windows guest desktop.
$ErrorActionPreference='Stop'
$root='C:\Users\baiwc\Downloads\Q3Practice'
$cfg=Get-Content -Raw (Join-Path $root 'bulk_config.json') | ConvertFrom-Json
$batch=Join-Path $root ('batches\'+$cfg.batch_id)
New-Item -ItemType Directory -Force $batch,(Join-Path $batch 'screenshots'),(Join-Path $batch 'runs'),(Join-Path $batch 'official_logs') | Out-Null
$started=Get-Date
$completed=@()
$phase='initializing'
$current=$null
function Write-Json($obj,$path){$tmp=$path+'.tmp';$obj | ConvertTo-Json -Depth 15 -Compress | Set-Content -Encoding UTF8 $tmp;Move-Item -Force $tmp $path}
function Append-Ledger($row,$path){
 # Guest-agent downloads can briefly hold a Windows file-sharing lock.
 # Retry opening only; never retry a write whose outcome could be partial.
 $bytes=[System.Text.Encoding]::UTF8.GetBytes(($row | ConvertTo-Json -Depth 10 -Compress)+[Environment]::NewLine)
 $stream=$null
 for($attempt=0;$attempt -lt 120;$attempt++){
  try{$stream=[System.IO.FileStream]::new($path,[System.IO.FileMode]::Append,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read);break}
  catch {
   $errorCause=$_.Exception;while($errorCause.InnerException){$errorCause=$errorCause.InnerException}
   $code=$errorCause.HResult -band 65535
   if($errorCause -isnot [System.IO.IOException] -or $code -notin @(32,33) -or $attempt -eq 119){throw}
   Start-Sleep -Milliseconds 250
  }
 }
 try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
}
function Status($state,$message=''){
 Write-Json @{status=$state;phase=$phase;message=$message;pid=$PID;batch_id=$cfg.batch_id;completed=$completed.Count;target=$cfg.schedule.Count;current=$current;updated_at=(Get-Date).ToString('o');session_started=$started.ToString('o');last_results=@($completed | Select-Object -Last 12)} (Join-Path $batch 'status.json')
}
try {
 Add-Type -AssemblyName System.Drawing,System.Windows.Forms,System.Runtime.WindowsRuntime
 Add-Type @'
using System;using System.Runtime.InteropServices;
public class Q3BulkUI {
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
 [Q3BulkUI]::SetProcessDpiAwarenessContext([IntPtr](-4)) | Out-Null
 [Q3BulkUI]::SetThreadExecutionState(2147483651) | Out-Null
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
  [Q3BulkUI]::SetForegroundWindow($hwnd) | Out-Null
  Start-Sleep -Milliseconds 100
  if([Q3BulkUI]::GetForegroundWindow() -ne $hwnd){throw 'Guest simulator not foreground'}
  $rect=New-Object Q3BulkUI+RECT
  [Q3BulkUI]::GetWindowRect($hwnd,[ref]$rect) | Out-Null
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
   $listText=View $tag 365 275 1185 65
   if($listText -match '开始问题3演练' -and $listText -match '全向干扰源'){return $listText}
  }
  throw "Q3 practice list guard timed out: $listText"
 }
 function Click($x,$y){
  Check-Window
  # All permitted controls are in the Q3 content area; sidebar/formal controls excluded.
  if($x -lt 600 -or $x -gt 1550 -or $y -lt 275 -or $y -gt 550){throw 'Click outside reviewed Q3 control area'}
  $screen=[System.Windows.Forms.SystemInformation]::VirtualScreen
  if(-not [Q3BulkUI]::ClickAt($x,$y,$screen.Left,$screen.Top,$screen.Width,$screen.Height)){throw 'Guest SendInput did not insert all click events'}
  Start-Sleep -Milliseconds 700
 }
 $ledger=Join-Path $batch 'completed.jsonl'
 if(Test-Path $ledger){$completed=@(Get-Content $ledger | Where-Object {$_} | ForEach-Object {$_ | ConvertFrom-Json})}
 $done=@{};foreach($row in $completed){$done[[int]$row.index]=$true}
 $newCount=0
 foreach($entry in $cfg.schedule){
  if($done.ContainsKey([int]$entry.index)){continue}
  if(Test-Path (Join-Path $batch 'STOP')){$phase='between_cases';Status 'stopped_by_marker';break}
  $current=$entry;$prefix=('{0:D4}_{1}' -f [int]$entry.index,$entry.method)
  $runDir=Join-Path $batch ('runs\'+$prefix)
  $resumeResult=Test-Path $runDir
  if($resumeResult){
   $savedResults=@(Get-ChildItem $runDir -Recurse -Filter result.json)
   if($savedResults.Count -ne 1){throw "Uncommitted attempt needs review: $runDir"}
   $saved=Get-Content -Raw -Encoding UTF8 $savedResults[0].FullName | ConvertFrom-Json
   if($saved.status -ne 'policy_completed_and_exited'){throw 'Prior policy attempt failed; preserve it for review'}
  }
  $phase='checking_practice_list';Status 'running'
  $resumeReady=($cfg.resume_ready_index -and [int]$cfg.resume_ready_index -eq [int]$entry.index)
  if(-not $resumeReady -and -not $resumeResult){
   $list=Wait-PracticeList ($prefix+'_list')
  }
  $logRoot='C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator'
  $before=@{}
  if($resumeResult){
   $assignment=Get-Content -Raw -Encoding UTF8 (Join-Path $runDir 'assignment.json') | ConvertFrom-Json
   if($assignment.prior_logs){foreach($p in $assignment.prior_logs){$before[$p]=$true}}
   else {Get-ChildItem $logRoot -Recurse -Filter 'practice-p3-*.jlog' | Where-Object {$_.LastWriteTime -lt [datetime]$assignment.started_at} | ForEach-Object {$before[$_.FullName]=$true}}
   $label=$assignment.case_label_ocr
  }else {Get-ChildItem $logRoot -Recurse -Filter 'practice-p3-*.jlog' | ForEach-Object {$before[$_.FullName]=$true}}
  if(-not $resumeResult){
  $phase='preparing_case';Status 'running'
  if(-not $resumeReady){Click 1433 309;Start-Sleep -Seconds 7}
  $ready=$false
  for($try=0;$try -lt 45;$try++){
   Start-Sleep -Seconds 2
   # This crop excludes the lower case row while the countdown banner is present.
   $top=View ($prefix+'_ready') 360 210 1200 145
   if($top -match '问题3演练' -and $top -match '尚未进入' -and $top -notmatch '倒计时|正在准备'){$ready=$true;break}
  }
  if(-not $ready){throw "Q3 ready guard timed out: $top"}
  $caseMatch=[regex]::Match($top,'[A-Z0-9]{4}[-一][A-Z0-9]{4}[-一][A-Z0-9]{4}[-一][A-Z0-9]{4}')
  $label=if($caseMatch.Success){$caseMatch.Value.Replace('一','-')}else{$prefix}
  New-Item -ItemType Directory $runDir | Out-Null
  Write-Json @{index=$entry.index;method=$entry.method;case_label_ocr=$label;ready_text=$top;started_at=(Get-Date).ToString('o');prior_logs=@($before.Keys)} (Join-Path $runDir 'assignment.json')
  $phase='running_policy';Status 'running'
  $argv=@((Join-Path $root 'bootstrap.py'),'--method',$entry.method,'--robot-id',$cfg.robot_id,'--case-label',$label,'--practice-confirmed','--output',$runDir)
  $proc=Start-Process -FilePath 'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe' -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $runDir 'stdout.log') -RedirectStandardError (Join-Path $runDir 'stderr.log')
  if(-not $proc.WaitForExit(1230000)){throw 'Policy did not exit within the official real-time window'}
  }
  $resultFiles=@(Get-ChildItem $runDir -Recurse -Filter result.json)
  if($resultFiles.Count -ne 1){throw 'Expected one policy result file'}
  $result=Get-Content -Raw -Encoding UTF8 $resultFiles[0].FullName | ConvertFrom-Json
  if($result.status -ne 'policy_completed_and_exited' -or -not $result.policy.complete_channel_certificate -or -not $result.exit_response.accepted){throw 'Policy failed or did not complete its certificate'}
  $phase='verifying_result';Status 'running'
  $verified=$false
  $dialogClosed=$false
  # A completed but uncommitted run may be resumed after its modal was closed.
  # Keep its original verified modal OCR and also check the live completed page.
  if($resumeResult){
   $modalCandidates=@((Join-Path $runDir 'completion_modal.json'),(Join-Path $batch ('screenshots\'+$prefix+'_done.png.json')))
   $modalCandidates+=@(Get-ChildItem (Join-Path $batch 'automation_events') -Recurse -Filter ($prefix+'_done.png.json') -ErrorAction SilentlyContinue | ForEach-Object {$_.FullName})
   $savedDialog=''
   foreach($candidate in $modalCandidates){
    if(-not (Test-Path $candidate)){continue}
    $candidateText=(Get-Content -Raw -Encoding UTF8 $candidate | ConvertFrom-Json).text
    if($candidateText -match '问题3演练.{0,3}完成' -and $candidateText -match '/exit' -and $candidateText -match '正常结束' -and $candidateText -match '日志已保存'){$savedDialog=$candidateText;break}
   }
   $page=View ($prefix+'_done_page') 360 210 1200 680
   $resumeLogs=@(Get-ChildItem $logRoot -Recurse -Filter 'practice-p3-*.jlog' | Where-Object {-not $before.ContainsKey($_.FullName)})
   $resumeCase=if($resumeLogs.Count -eq 1){[regex]::Match($resumeLogs[0].Name,'([A-Z0-9]{4}(?:-[A-Z0-9]{4}){3})\.jlog$').Groups[1].Value}else{''}
   $pageOmni=[regex]::Match($page,'全向(\d+)个');$pageDirectional=[regex]::Match($page,'定向(\d+)个')
   $pageCountMatches=$pageOmni.Success -and $pageDirectional.Success -and ([int]$pageOmni.Groups[1].Value+[int]$pageDirectional.Groups[1].Value) -eq [int]$result.client_state.cleared_count
   if($savedDialog -and $resumeCase -and $page -match [regex]::Escape($resumeCase) -and $pageCountMatches -and $page -match '问题3演练' -and $page -match '返回演练' -and $page -match '行为日志已保存'){
    $dialog=$savedDialog;$verified=$true;$dialogClosed=$true
   }
  }
  for($try=0;-not $verified -and $try -lt 30;$try++){
   Start-Sleep -Milliseconds 500
   $dialog=View ($prefix+'_done') 610 345 500 225
   if($dialog -match '问题3演练.{0,3}完成' -and $dialog -match '/exit' -and $dialog -match '正常结束' -and $dialog -match '日志已保存'){$verified=$true;break}
  }
  if(-not $verified){throw "Q3 completion guard failed: $dialog"}
  $countMatch=[regex]::Match($dialog,'(?:干扰源|干扰源数量)(\d+)个')
  $omni=[regex]::Match($dialog,'全向(\d+)个')
  $directional=[regex]::Match($dialog,'定向(\d+)个')
  if($countMatch.Success){$total=[int]$countMatch.Groups[1].Value}
  elseif($omni.Success -and $directional.Success){$total=[int]$omni.Groups[1].Value+[int]$directional.Groups[1].Value}
  else {throw "Could not read official total source count: $dialog"}
  if($omni.Success -and $directional.Success -and $total -ne ([int]$omni.Groups[1].Value+[int]$directional.Groups[1].Value)){throw 'Conflicting source counts in OCR'}
  if($total -ne [int]$result.client_state.cleared_count){throw 'Successful clears differ from official displayed source count'}
  if(-not (Test-Path (Join-Path $runDir 'completion_modal.json'))){Write-Json @{text=$dialog;total=$total;verified_at=(Get-Date).ToString('o')} (Join-Path $runDir 'completion_modal.json')}
  $newLogs=@(Get-ChildItem $logRoot -Recurse -Filter 'practice-p3-*.jlog' | Where-Object {-not $before.ContainsKey($_.FullName)})
  if($newLogs.Count -ne 1 -or $newLogs[0].Length -lt 1){throw 'Expected exactly one new official Q3 behavior log'}
  $case=[regex]::Match($newLogs[0].Name,'([A-Z0-9]{4}(?:-[A-Z0-9]{4}){3})\.jlog$').Groups[1].Value
  if(-not $case){throw 'Official log has no case code'}
  Copy-Item $newLogs[0].FullName (Join-Path $batch ('official_logs\'+$newLogs[0].Name))
  $failures=0;foreach($source in $result.client_state.sources.PSObject.Properties){$failures+=[int]$source.Value.failed_clear_count}
  $row=@{index=[int]$entry.index;block=[int]$entry.block;method=$entry.method;case=$case;case_label_ocr=$label;total=$total;cleared=[int]$result.client_state.cleared_count;virtual_s=[double]$result.client_state.virtual_time_s;seconds_per_source=[double]$result.seconds_per_accepted_clear;wall_s=[double]$result.wall_seconds;nonrequest_s=[double]$result.policy.nonrequest_wall_s;failed_clear_attempts=$failures;result_path=$resultFiles[0].FullName;official_log=$newLogs[0].Name;official_log_sha256=(Get-FileHash $newLogs[0].FullName -Algorithm SHA256).Hash.ToLower();completed_at=(Get-Date).ToString('o');ui_text=$dialog}
  Append-Ledger $row $ledger
  $completed+=,$row;$done[[int]$entry.index]=$true;$newCount++
  $phase='returning_to_list';Status 'running'
  if(-not $dialogClosed){Click 1018 531}
  $return=View ($prefix+'_return') 1390 278 165 62
  if($return -notmatch '返回演练'){throw "Return guard failed: $return"}
  Click 1466 308
  $null=Wait-PracticeList ($prefix+'_return_list')
  if($cfg.run_limit -and $newCount -ge [int]$cfg.run_limit){$phase='between_cases';Status 'paused_after_limit';break}
 }
 if($completed.Count -eq $cfg.schedule.Count){$phase='finished';$current=$null;Status 'completed'}
} catch {Status 'error' ($_.Exception.ToString()+' '+$_.InvocationInfo.PositionMessage)}
finally {[Q3BulkUI]::SetThreadExecutionState(2147483648) | Out-Null}
