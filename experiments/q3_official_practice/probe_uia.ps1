$ErrorActionPreference='Stop'
$out='C:\Users\baiwc\Downloads\Q3Practice\uia_probe.json'
try {
  Add-Type -AssemblyName UIAutomationClient
  Add-Type -AssemblyName UIAutomationTypes
  $sim=Get-Process -Name jammers-simulator | Select-Object -First 1
  $root=[System.Windows.Automation.AutomationElement]::RootElement
  $windows=$root.FindAll([System.Windows.Automation.TreeScope]::Children,[System.Windows.Automation.Condition]::TrueCondition)
  $matches=@()
  foreach($w in $windows) {
    if($w.Current.ProcessId -eq $sim.Id) {
      $nodes=$w.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)
      foreach($n in $nodes) {
        $matches+=@{name=$n.Current.Name;type=$n.Current.ControlType.ProgrammaticName;id=$n.Current.AutomationId;enabled=$n.Current.IsEnabled;patterns=@($n.GetSupportedPatterns() | ForEach-Object {$_.ProgrammaticName})}
      }
    }
  }
  @{session=(Get-Process -Id $PID).SessionId;user=[Environment]::UserName;sim_pid=$sim.Id;sim_session=$sim.SessionId;sim_handle=[long]$sim.MainWindowHandle;window_count=$windows.Count;nodes=$matches} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $out
} catch { @{error=$_.Exception.ToString();session=(Get-Process -Id $PID).SessionId} | ConvertTo-Json | Set-Content -Encoding UTF8 $out }
