param(
    [string]$OutputDir = "figures"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ood = Import-Csv "neural_ood_binary_metrics.csv"
$mc = Import-Csv "neural_multiclass_metrics.csv"

$labels = @{
    "primary__compressed_plus_losses" = "Primary: compressed + losses"
    "primary__compressed_only" = "Primary: compressed only"
    "primary__all_compressed_features" = "Primary: all compressed features"
    "no_time_adv__compressed_plus_losses" = "No time adversary"
    "no_student__compressed_plus_losses" = "No student loss"
}

$order = @(
    "primary__compressed_plus_losses",
    "primary__compressed_only",
    "primary__all_compressed_features",
    "no_time_adv__compressed_plus_losses",
    "no_student__compressed_plus_losses"
)

function To-Double($value) {
    return [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
}

function Format-Pct($value) {
    return (To-Double $value).ToString("P1", [Globalization.CultureInfo]::InvariantCulture)
}

function Escape-Xml($text) {
    return [System.Security.SecurityElement]::Escape([string]$text)
}

function New-BarSvg {
    param(
        [string]$Title,
        [string]$Subtitle,
        [object[]]$Rows,
        [string[]]$Metrics,
        [hashtable]$MetricLabels,
        [string]$Path,
        [double]$YMin = 0.0,
        [double]$YMax = 1.0
    )

    $width = 1320
    $height = 760
    $left = 260
    $right = 70
    $top = 105
    $bottom = 185
    $plotW = $width - $left - $right
    $plotH = $height - $top - $bottom
    $groupGap = 34
    $barGap = 6
    $colors = @("#2563eb", "#059669", "#dc2626", "#7c3aed")
    $groupW = ($plotW - ($groupGap * ($Rows.Count - 1))) / $Rows.Count
    $barW = [Math]::Min(42, ($groupW - ($barGap * ($Metrics.Count - 1))) / $Metrics.Count)
    $svg = New-Object System.Collections.Generic.List[string]

    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""$width"" height=""$height"" viewBox=""0 0 $width $height"">")
    $svg.Add("<rect width=""100%"" height=""100%"" fill=""#ffffff""/>")
    $svg.Add("<text x=""$left"" y=""42"" font-family=""Arial, sans-serif"" font-size=""28"" font-weight=""700"" fill=""#111827"">$(Escape-Xml $Title)</text>")
    $svg.Add("<text x=""$left"" y=""72"" font-family=""Arial, sans-serif"" font-size=""15"" fill=""#4b5563"">$(Escape-Xml $Subtitle)</text>")

    for ($i = 0; $i -le 5; $i++) {
        $value = $YMin + (($YMax - $YMin) * $i / 5.0)
        $y = $top + $plotH - (($value - $YMin) / ($YMax - $YMin) * $plotH)
        $label = $value.ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
        $svg.Add("<line x1=""$left"" y1=""$y"" x2=""$($left + $plotW)"" y2=""$y"" stroke=""#e5e7eb"" stroke-width=""1""/>")
        $svg.Add("<text x=""$($left - 14)"" y=""$($y + 5)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""13"" fill=""#6b7280"">$label</text>")
    }

    $svg.Add("<line x1=""$left"" y1=""$top"" x2=""$left"" y2=""$($top + $plotH)"" stroke=""#9ca3af""/>")
    $svg.Add("<line x1=""$left"" y1=""$($top + $plotH)"" x2=""$($left + $plotW)"" y2=""$($top + $plotH)"" stroke=""#9ca3af""/>")

    for ($r = 0; $r -lt $Rows.Count; $r++) {
        $row = $Rows[$r]
        $groupX = $left + ($r * ($groupW + $groupGap))
        $label = $labels[$row.experiment]
        for ($m = 0; $m -lt $Metrics.Count; $m++) {
            $metric = $Metrics[$m]
            $value = To-Double $row.$metric
            $barH = [Math]::Max(0, ($value - $YMin) / ($YMax - $YMin) * $plotH)
            $x = $groupX + (($groupW - ($Metrics.Count * $barW + ($Metrics.Count - 1) * $barGap)) / 2) + $m * ($barW + $barGap)
            $y = $top + $plotH - $barH
            $color = $colors[$m % $colors.Count]
            $svg.Add("<rect x=""$x"" y=""$y"" width=""$barW"" height=""$barH"" rx=""2"" fill=""$color""/>")
            $svg.Add("<text x=""$($x + $barW / 2)"" y=""$($y - 7)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""11"" fill=""#111827"">$($value.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture))</text>")
        }
        $parts = $label -split ": "
        $cx = $groupX + ($groupW / 2)
        $svg.Add("<text x=""$cx"" y=""$($top + $plotH + 34)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""14"" font-weight=""600"" fill=""#111827"">$(Escape-Xml $parts[0])</text>")
        if ($parts.Count -gt 1) {
            $svg.Add("<text x=""$cx"" y=""$($top + $plotH + 55)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""13"" fill=""#4b5563"">$(Escape-Xml $parts[1])</text>")
        }
    }

    $legendX = $left
    $legendY = $height - 64
    for ($m = 0; $m -lt $Metrics.Count; $m++) {
        $x = $legendX + ($m * 245)
        $svg.Add("<rect x=""$x"" y=""$legendY"" width=""15"" height=""15"" fill=""$($colors[$m % $colors.Count])""/>")
        $svg.Add("<text x=""$($x + 23)"" y=""$($legendY + 13)"" font-family=""Arial, sans-serif"" font-size=""14"" fill=""#111827"">$(Escape-Xml $MetricLabels[$Metrics[$m]])</text>")
    }

    $svg.Add("</svg>")
    [System.IO.File]::WriteAllText($Path, ($svg -join "`n"), [System.Text.UTF8Encoding]::new($false))
}

function New-DeltaSvg {
    param(
        [object[]]$Rows,
        [string]$Path
    )

    $baseline = $Rows | Where-Object { $_.experiment -eq "primary__compressed_plus_losses" } | Select-Object -First 1
    $others = $Rows | Where-Object { $_.experiment -ne "primary__compressed_plus_losses" }
    $metrics = @("attack_recall", "macro_f1", "benign_specificity")
    $metricLabels = @{
        "attack_recall" = "OOD attack recall"
        "macro_f1" = "OOD macro F1"
        "benign_specificity" = "Benign specificity"
    }
    $summary = foreach ($row in $Rows) {
        [pscustomobject]@{
            experiment = $row.experiment
            display_name = $labels[$row.experiment]
            ood_attack_recall = To-Double $row.attack_recall
            ood_macro_f1 = To-Double $row.macro_f1
            ood_bal_acc = To-Double $row.bal_acc
            benign_specificity = To-Double $row.benign_specificity
            unseen_recall = To-Double $row.unseen_recall
            delta_attack_recall_vs_primary = (To-Double $row.attack_recall) - (To-Double $baseline.attack_recall)
            delta_macro_f1_vs_primary = (To-Double $row.macro_f1) - (To-Double $baseline.macro_f1)
        }
    }
    $summary | Export-Csv (Join-Path $OutputDir "ablation_summary.csv") -NoTypeInformation

    $width = 1320
    $height = 700
    $left = 300
    $right = 80
    $top = 96
    $rowH = 102
    $axisX = $left + 470
    $scale = 430
    $colors = @{
        "attack_recall" = "#2563eb"
        "macro_f1" = "#059669"
        "benign_specificity" = "#dc2626"
    }
    $svg = New-Object System.Collections.Generic.List[string]
    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""$width"" height=""$height"" viewBox=""0 0 $width $height"">")
    $svg.Add("<rect width=""100%"" height=""100%"" fill=""#ffffff""/>")
    $svg.Add("<text x=""$left"" y=""42"" font-family=""Arial, sans-serif"" font-size=""28"" font-weight=""700"" fill=""#111827"">Ablation impact relative to primary compressed + losses</text>")
    $svg.Add("<text x=""$left"" y=""72"" font-family=""Arial, sans-serif"" font-size=""15"" fill=""#4b5563"">Negative values show how much performance drops when a component or feature view is removed.</text>")
    $svg.Add("<line x1=""$axisX"" y1=""$top"" x2=""$axisX"" y2=""$($top + $rowH * $others.Count)"" stroke=""#111827"" stroke-width=""1.5""/>")
    foreach ($tick in @(-1.0, -0.75, -0.5, -0.25, 0.0)) {
        $x = $axisX + ($tick * $scale)
        $svg.Add("<line x1=""$x"" y1=""$top"" x2=""$x"" y2=""$($top + $rowH * $others.Count)"" stroke=""#e5e7eb""/>")
        $svg.Add("<text x=""$x"" y=""$($top + $rowH * $others.Count + 28)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""13"" fill=""#6b7280"">$tick</text>")
    }
    $idx = 0
    foreach ($row in $others) {
        $y0 = $top + $idx * $rowH + 28
        $svg.Add("<text x=""$left"" y=""$($y0 + 10)"" font-family=""Arial, sans-serif"" font-size=""16"" font-weight=""600"" fill=""#111827"">$(Escape-Xml $labels[$row.experiment])</text>")
        for ($m = 0; $m -lt $metrics.Count; $m++) {
            $metric = $metrics[$m]
            $delta = (To-Double $row.$metric) - (To-Double $baseline.$metric)
            $x = $axisX + ($delta * $scale)
            $y = $y0 + ($m * 22)
            $w = [Math]::Abs($axisX - $x)
            $barX = [Math]::Min($x, $axisX)
            $svg.Add("<rect x=""$barX"" y=""$($y - 13)"" width=""$w"" height=""16"" rx=""2"" fill=""$($colors[$metric])""/>")
            $svg.Add("<text x=""$($barX - 8)"" y=""$y"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""12"" fill=""#111827"">$($delta.ToString("+0.000;-0.000;0.000", [Globalization.CultureInfo]::InvariantCulture))</text>")
        }
        $idx += 1
    }
    $legendY = $height - 64
    for ($m = 0; $m -lt $metrics.Count; $m++) {
        $x = $left + ($m * 255)
        $metric = $metrics[$m]
        $svg.Add("<rect x=""$x"" y=""$legendY"" width=""15"" height=""15"" fill=""$($colors[$metric])""/>")
        $svg.Add("<text x=""$($x + 23)"" y=""$($legendY + 13)"" font-family=""Arial, sans-serif"" font-size=""14"" fill=""#111827"">$(Escape-Xml $metricLabels[$metric])</text>")
    }
    $svg.Add("</svg>")
    [System.IO.File]::WriteAllText($Path, ($svg -join "`n"), [System.Text.UTF8Encoding]::new($false))
}

$orderedOod = foreach ($name in $order) { $ood | Where-Object { $_.experiment -eq $name } | Select-Object -First 1 }
$orderedMc = foreach ($name in $order) { $mc | Where-Object { $_.experiment -eq $name } | Select-Object -First 1 }

New-BarSvg `
    -Title "Final OOD attack detection across ablations" `
    -Subtitle "Compressed losses carry the unseen-attack signal; removing them sharply reduces recall and macro F1." `
    -Rows $orderedOod `
    -Metrics @("attack_recall", "unseen_recall", "macro_f1", "benign_specificity") `
    -MetricLabels @{
        "attack_recall" = "Attack recall"
        "unseen_recall" = "Unseen recall"
        "macro_f1" = "Macro F1"
        "benign_specificity" = "Benign specificity"
    } `
    -Path (Join-Path $OutputDir "ood_ablation_metrics.svg")

New-BarSvg `
    -Title "In-distribution multiclass performance across ablations" `
    -Subtitle "All compressed features preserve in-distribution accuracy but fail on final OOD detection." `
    -Rows $orderedMc `
    -Metrics @("bal_acc", "macro_f1", "hier_f1", "subtype_f1") `
    -MetricLabels @{
        "bal_acc" = "Balanced accuracy"
        "macro_f1" = "Macro F1"
        "hier_f1" = "Hierarchical F1"
        "subtype_f1" = "Subtype F1"
    } `
    -Path (Join-Path $OutputDir "multiclass_ablation_metrics.svg") `
    -YMin 0.75

New-BarSvg `
    -Title "Precision-recall operating point for final OOD detection" `
    -Subtitle "The primary compressed+losses run is the only setting with high attack recall and high attack precision." `
    -Rows $orderedOod `
    -Metrics @("attack_precision", "attack_recall", "benign_precision", "benign_recall") `
    -MetricLabels @{
        "attack_precision" = "Attack precision"
        "attack_recall" = "Attack recall"
        "benign_precision" = "Benign precision"
        "benign_recall" = "Benign recall"
    } `
    -Path (Join-Path $OutputDir "ood_precision_recall.svg")

New-DeltaSvg -Rows $orderedOod -Path (Join-Path $OutputDir "ablation_delta_vs_primary.svg")

Write-Host "wrote $(Join-Path $OutputDir 'ood_ablation_metrics.svg')"
Write-Host "wrote $(Join-Path $OutputDir 'multiclass_ablation_metrics.svg')"
Write-Host "wrote $(Join-Path $OutputDir 'ood_precision_recall.svg')"
Write-Host "wrote $(Join-Path $OutputDir 'ablation_delta_vs_primary.svg')"
Write-Host "wrote $(Join-Path $OutputDir 'ablation_summary.csv')"
