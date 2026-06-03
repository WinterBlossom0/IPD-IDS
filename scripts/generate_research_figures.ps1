param(
    [string]$OutputDir = "figures"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ood = Import-Csv "neural_ood_binary_metrics.csv"
$mc = Import-Csv "neural_multiclass_metrics.csv"

$order = @(
    "primary__compressed_plus_losses",
    "primary__compressed_only",
    "primary__all_compressed_features",
    "no_time_adv__compressed_plus_losses",
    "no_student__compressed_plus_losses"
)

$codes = @{
    "primary__compressed_plus_losses" = "CL"
    "primary__compressed_only" = "C"
    "primary__all_compressed_features" = "ACF"
    "no_time_adv__compressed_plus_losses" = "-T"
    "no_student__compressed_plus_losses" = "-S"
}

$labels = @{
    "primary__compressed_plus_losses" = "comp+loss"
    "primary__compressed_only" = "comp only"
    "primary__all_compressed_features" = "all comp"
    "no_time_adv__compressed_plus_losses" = "no time"
    "no_student__compressed_plus_losses" = "no student"
}

$codeColors = @{
    "CL" = "#1f4e79"
    "C" = "#b45f06"
    "ACF" = "#38761d"
    "-T" = "#674ea7"
    "-S" = "#990000"
}

function D($value) {
    return [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
}

function Num($value, [string]$format = "0.00") {
    return ([double]$value).ToString($format, [Globalization.CultureInfo]::InvariantCulture)
}

function Pct0($value) {
    return (([double]$value) * 100.0).ToString("0", [Globalization.CultureInfo]::InvariantCulture)
}

function Pct1($value) {
    return (([double]$value) * 100.0).ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
}

function Esc($text) {
    return [System.Security.SecurityElement]::Escape([string]$text)
}

function Write-Svg($path, [System.Collections.Generic.List[string]]$svg) {
    $svg.Add("</svg>")
    [System.IO.File]::WriteAllText($path, ($svg -join "`n"), [System.Text.UTF8Encoding]::new($false))
}

function XPos([double]$value, [double]$min, [double]$max, [double]$left, [double]$plotW) {
    $t = ($value - $min) / ($max - $min)
    $t = [Math]::Max(0.0, [Math]::Min(1.0, $t))
    return $left + $t * $plotW
}

function Add-Header([System.Collections.Generic.List[string]]$svg, [string]$title, [string]$subtitle) {
    $svg.Add("<rect width=""100%"" height=""100%"" fill=""#ffffff""/>")
    $svg.Add("<text x=""12"" y=""17"" font-family=""Arial, sans-serif"" font-size=""11"" font-weight=""700"" fill=""#111827"">$(Esc $title)</text>")
    $svg.Add("<text x=""12"" y=""30"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#4b5563"">$(Esc $subtitle)</text>")
}

function Add-CodeKey([System.Collections.Generic.List[string]]$svg, [int]$x, [int]$y) {
    $parts = @(
        @("CL", "comp+loss"),
        @("C", "comp only"),
        @("ACF", "all comp"),
        @("-T", "no time"),
        @("-S", "no student")
    )
    $cursor = $x
    foreach ($part in $parts) {
        $code = $part[0]
        $label = $part[1]
        $svg.Add("<text x=""$cursor"" y=""$y"" font-family=""Arial, sans-serif"" font-size=""7.2"" font-weight=""700"" fill=""$($codeColors[$code])"">$(Esc $code)</text>")
        $cursor += 18 + ($code.Length * 3)
        $svg.Add("<text x=""$cursor"" y=""$y"" font-family=""Arial, sans-serif"" font-size=""7.2"" fill=""#6b7280"">$(Esc $label)</text>")
        $cursor += 42
    }
}

function OrderedRows($rows) {
    foreach ($name in $order) {
        $rows | Where-Object { $_.experiment -eq $name } | Select-Object -First 1
    }
}

function New-TwoMetricBars {
    param(
        [string]$Title,
        [string]$Subtitle,
        [object[]]$Rows,
        [string]$MetricA,
        [string]$MetricB,
        [string]$LabelA,
        [string]$LabelB,
        [string]$Path,
        [double]$XMin = 0.0,
        [double]$XMax = 1.0,
        [double[]]$Ticks = @(0.0, 0.25, 0.5, 0.75, 1.0)
    )

    $w = 345
    $h = 235
    $left = 82
    $right = 18
    $top = 58
    $plotW = $w - $left - $right
    $rowGap = 27
    $barH = 5.5
    $blue = "#2563eb"
    $green = "#059669"
    $svg = [System.Collections.Generic.List[string]]::new()
    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""3.45in"" height=""2.35in"" viewBox=""0 0 $w $h"">")
    Add-Header $svg $Title $Subtitle

    foreach ($tick in $Ticks) {
        $x = XPos $tick $XMin $XMax $left $plotW
        $svg.Add("<line x1=""$x"" y1=""50"" x2=""$x"" y2=""$($top + ($Rows.Count - 1) * $rowGap + 24)"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $tickLabel = if ($XMax -le 1.01) { Pct0 $tick } else { Num $tick "0.0" }
        $svg.Add("<text x=""$x"" y=""$($h - 17)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$tickLabel</text>")
    }
    $svg.Add("<line x1=""$left"" y1=""$($top + ($Rows.Count - 1) * $rowGap + 24)"" x2=""$($left + $plotW)"" y2=""$($top + ($Rows.Count - 1) * $rowGap + 24)"" stroke=""#9ca3af"" stroke-width=""0.8""/>")

    $legendY = 45
    $svg.Add("<rect x=""$left"" y=""$($legendY - 6)"" width=""9"" height=""5"" rx=""2"" fill=""$blue""/>")
    $svg.Add("<text x=""$($left + 14)"" y=""$legendY"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">$(Esc $LabelA)</text>")
    $svg.Add("<rect x=""$($left + 94)"" y=""$($legendY - 6)"" width=""9"" height=""5"" rx=""2"" fill=""$green""/>")
    $svg.Add("<text x=""$($left + 108)"" y=""$legendY"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">$(Esc $LabelB)</text>")

    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $row = $Rows[$i]
        $code = $codes[$row.experiment]
        $label = "$code  $($labels[$row.experiment])"
        $y = $top + $i * $rowGap
        $svg.Add("<text x=""74"" y=""$($y + 12)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""7.8"" font-weight=""700"" fill=""$($codeColors[$code])"">$(Esc $label)</text>")
        $svg.Add("<line x1=""$left"" y1=""$($y + 4)"" x2=""$($left + $plotW)"" y2=""$($y + 4)"" stroke=""#f3f4f6"" stroke-width=""0.8""/>")

        $a = D $row.$MetricA
        $b = D $row.$MetricB
        $xa = XPos $a $XMin $XMax $left $plotW
        $xb = XPos $b $XMin $XMax $left $plotW
        $ya = $y + 5
        $yb = $y + 15

        $svg.Add("<rect x=""$left"" y=""$ya"" width=""$([Math]::Max(1, $xa - $left))"" height=""$barH"" rx=""2.5"" fill=""$blue""/>")
        $svg.Add("<rect x=""$left"" y=""$yb"" width=""$([Math]::Max(1, $xb - $left))"" height=""$barH"" rx=""2.5"" fill=""$green""/>")

        $aLabelX = if ($xa -gt ($left + $plotW - 16)) { $xa - 18 } else { $xa + 4 }
        $bLabelX = if ($xb -gt ($left + $plotW - 16)) { $xb - 18 } else { $xb + 4 }
        $aAnchor = if ($xa -gt ($left + $plotW - 16)) { "end" } else { "start" }
        $bAnchor = if ($xb -gt ($left + $plotW - 16)) { "end" } else { "start" }
        $svg.Add("<text x=""$aLabelX"" y=""$($ya + 5.3)"" text-anchor=""$aAnchor"" font-family=""Arial, sans-serif"" font-size=""6.6"" font-weight=""700"" fill=""#111827"">$(Pct0 $a)</text>")
        $svg.Add("<text x=""$bLabelX"" y=""$($yb + 5.3)"" text-anchor=""$bAnchor"" font-family=""Arial, sans-serif"" font-size=""6.6"" font-weight=""700"" fill=""#111827"">$(Pct0 $b)</text>")
    }

    $svg.Add("<text x=""$($left + $plotW / 2)"" y=""$($h - 4)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">Metric value (%)</text>")
    Write-Svg $Path $svg
}

function New-OperatingPoint {
    param(
        [object[]]$Rows,
        [string]$Path
    )

    $w = 345
    $h = 235
    $left = 52
    $right = 18
    $top = 48
    $plotW = $w - $left - $right
    $plotH = 135
    $yMax = 0.17
    $svg = [System.Collections.Generic.List[string]]::new()
    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""3.45in"" height=""2.35in"" viewBox=""0 0 $w $h"">")
    Add-Header $svg "OOD operating point" "High recall is only useful when the false-positive rate stays low."

    foreach ($tick in @(0.0, 0.25, 0.5, 0.75, 1.0)) {
        $x = XPos $tick 0 1 $left $plotW
        $svg.Add("<line x1=""$x"" y1=""$top"" x2=""$x"" y2=""$($top + $plotH)"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $svg.Add("<text x=""$x"" y=""$($top + $plotH + 14)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$(Pct0 $tick)</text>")
    }
    foreach ($tick in @(0.0, 0.05, 0.10, 0.15)) {
        $y = $top + $plotH - ($tick / $yMax * $plotH)
        $svg.Add("<line x1=""$left"" y1=""$y"" x2=""$($left + $plotW)"" y2=""$y"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $svg.Add("<text x=""$($left - 7)"" y=""$($y + 2.6)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$((100 * $tick).ToString('0', [Globalization.CultureInfo]::InvariantCulture))</text>")
    }

    $svg.Add("<line x1=""$left"" y1=""$top"" x2=""$left"" y2=""$($top + $plotH)"" stroke=""#9ca3af"" stroke-width=""0.8""/>")
    $svg.Add("<line x1=""$left"" y1=""$($top + $plotH)"" x2=""$($left + $plotW)"" y2=""$($top + $plotH)"" stroke=""#9ca3af"" stroke-width=""0.8""/>")

    foreach ($row in $Rows) {
        $code = $codes[$row.experiment]
        $xVal = D $row.attack_recall
        $fpr = 1.0 - (D $row.benign_specificity)
        $x = XPos $xVal 0 1 $left $plotW
        $y = $top + $plotH - ([Math]::Min($yMax, $fpr) / $yMax * $plotH)
        $fill = $codeColors[$code]
        $svg.Add("<circle cx=""$x"" cy=""$y"" r=""4.3"" fill=""$fill"" stroke=""#ffffff"" stroke-width=""1.2""/>")

        $lx = $x + 6
        $ly = $y - 6
        if ($code -eq "CL") { $lx = $x - 20; $ly = $y + 12 }
        if ($code -eq "C") { $lx = $x + 7; $ly = $y - 9 }
        if ($code -eq "ACF") { $lx = $x + 7; $ly = $y + 12 }
        if ($code -eq "-S") { $lx = $x + 7; $ly = $y - 3 }
        if ($code -eq "-T") { $lx = $x + 7; $ly = $y + 3 }
        $svg.Add("<text x=""$lx"" y=""$ly"" font-family=""Arial, sans-serif"" font-size=""7.8"" font-weight=""700"" fill=""$fill"">$(Esc $code)</text>")
    }

    $svg.Add("<text x=""$($left + $plotW / 2)"" y=""$($h - 5)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">Attack recall (%)</text>")
    $svg.Add("<text x=""12"" y=""$($top + 65)"" transform=""rotate(-90 12,$($top + 65))"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">False-positive rate (%)</text>")
    Write-Svg $Path $svg
}

function New-RetentionFigure {
    param(
        [object[]]$Rows,
        [string]$Path
    )

    $baseline = $Rows | Where-Object { $_.experiment -eq "primary__compressed_plus_losses" } | Select-Object -First 1
    $others = $Rows | Where-Object { $_.experiment -ne "primary__compressed_plus_losses" }
    $w = 345
    $h = 235
    $left = 82
    $plotW = 245
    $top = 60
    $rowGap = 33
    $barH = 6
    $blue = "#2563eb"
    $green = "#059669"
    $svg = [System.Collections.Generic.List[string]]::new()
    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""3.45in"" height=""2.35in"" viewBox=""0 0 $w $h"">")
    Add-Header $svg "Metric retained after ablation" "Bars show the percentage of CL performance retained."

    foreach ($tick in @(0.0, 0.25, 0.5, 0.75, 1.0)) {
        $x = XPos $tick 0 1 $left $plotW
        $svg.Add("<line x1=""$x"" y1=""50"" x2=""$x"" y2=""$($top + ($others.Count - 1) * $rowGap + 25)"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $svg.Add("<text x=""$x"" y=""$($h - 17)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$(Pct0 $tick)</text>")
    }
    $svg.Add("<rect x=""$left"" y=""43"" width=""9"" height=""5"" rx=""2"" fill=""$blue""/>")
    $svg.Add("<text x=""$($left + 14)"" y=""49"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">Recall retained</text>")
    $svg.Add("<rect x=""$($left + 100)"" y=""43"" width=""9"" height=""5"" rx=""2"" fill=""$green""/>")
    $svg.Add("<text x=""$($left + 114)"" y=""49"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">F1 retained</text>")

    for ($i = 0; $i -lt $others.Count; $i++) {
        $row = $others[$i]
        $code = $codes[$row.experiment]
        $label = "$code  $($labels[$row.experiment])"
        $y = $top + $i * $rowGap
        $svg.Add("<text x=""74"" y=""$($y + 14)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""7.8"" font-weight=""700"" fill=""$($codeColors[$code])"">$(Esc $label)</text>")
        $rec = if ((D $baseline.attack_recall) -eq 0) { 0 } else { (D $row.attack_recall) / (D $baseline.attack_recall) }
        $f1 = if ((D $baseline.macro_f1) -eq 0) { 0 } else { (D $row.macro_f1) / (D $baseline.macro_f1) }
        $xr = XPos $rec 0 1 $left $plotW
        $xf = XPos $f1 0 1 $left $plotW
        $svg.Add("<rect x=""$left"" y=""$($y + 4)"" width=""$([Math]::Max(1, $xr - $left))"" height=""$barH"" rx=""2.5"" fill=""$blue""/>")
        $svg.Add("<rect x=""$left"" y=""$($y + 16)"" width=""$([Math]::Max(1, $xf - $left))"" height=""$barH"" rx=""2.5"" fill=""$green""/>")
        $svg.Add("<text x=""$($xr + 4)"" y=""$($y + 10)"" font-family=""Arial, sans-serif"" font-size=""6.6"" font-weight=""700"" fill=""#111827"">$(Pct0 $rec)</text>")
        $svg.Add("<text x=""$($xf + 4)"" y=""$($y + 22)"" font-family=""Arial, sans-serif"" font-size=""6.6"" font-weight=""700"" fill=""#111827"">$(Pct0 $f1)</text>")
    }
    $svg.Add("<text x=""$($left + $plotW / 2)"" y=""$($h - 4)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7.4"" fill=""#374151"">Share of CL performance retained (%)</text>")
    Write-Svg $Path $svg
}

$orderedOod = @(OrderedRows $ood)
$orderedMc = @(OrderedRows $mc)
$baselineOod = $orderedOod | Where-Object { $_.experiment -eq "primary__compressed_plus_losses" } | Select-Object -First 1

$summary = foreach ($row in $orderedOod) {
    $matchingMc = $orderedMc | Where-Object { $_.experiment -eq $row.experiment } | Select-Object -First 1
    [pscustomobject]@{
        experiment = $row.experiment
        code = $codes[$row.experiment]
        display_name = $labels[$row.experiment]
        ood_attack_recall = D $row.attack_recall
        ood_macro_f1 = D $row.macro_f1
        ood_bal_acc = D $row.bal_acc
        benign_specificity = D $row.benign_specificity
        false_positive_rate = 1.0 - (D $row.benign_specificity)
        unseen_recall = D $row.unseen_recall
        in_dist_hier_f1 = D $matchingMc.hier_f1
        in_dist_macro_f1 = D $matchingMc.macro_f1
        recall_retained_vs_cl = if ((D $baselineOod.attack_recall) -eq 0) { 0 } else { (D $row.attack_recall) / (D $baselineOod.attack_recall) }
        macro_f1_retained_vs_cl = if ((D $baselineOod.macro_f1) -eq 0) { 0 } else { (D $row.macro_f1) / (D $baselineOod.macro_f1) }
    }
}
$summary | Export-Csv (Join-Path $OutputDir "ablation_summary.csv") -NoTypeInformation

New-TwoMetricBars `
    -Title "Final OOD detection" `
    -Subtitle "Only compressed+loss features retain high unseen-attack detection." `
    -Rows $orderedOod `
    -MetricA "attack_recall" `
    -MetricB "macro_f1" `
    -LabelA "Attack recall" `
    -LabelB "Macro F1" `
    -Path (Join-Path $OutputDir "ood_ablation_metrics.svg")

New-TwoMetricBars `
    -Title "In-distribution classification" `
    -Subtitle "High ID scores alone do not imply final OOD robustness." `
    -Rows $orderedMc `
    -MetricA "hier_f1" `
    -MetricB "macro_f1" `
    -LabelA "Hier. F1" `
    -LabelB "Macro F1" `
    -Path (Join-Path $OutputDir "multiclass_ablation_metrics.svg") `
    -XMin 0.75 `
    -XMax 1.0 `
    -Ticks @(0.75, 0.85, 0.95, 1.0)

New-OperatingPoint -Rows $orderedOod -Path (Join-Path $OutputDir "ood_precision_recall.svg")
New-RetentionFigure -Rows $orderedOod -Path (Join-Path $OutputDir "ablation_delta_vs_primary.svg")

Write-Host "wrote publication-style SVG figures to $OutputDir"
