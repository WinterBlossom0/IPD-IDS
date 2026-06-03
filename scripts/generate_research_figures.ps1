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

$shortLabels = @{
    "primary__compressed_plus_losses" = "CL"
    "primary__compressed_only" = "C"
    "primary__all_compressed_features" = "ACF"
    "no_time_adv__compressed_plus_losses" = "-T"
    "no_student__compressed_plus_losses" = "-S"
}

$displayLabels = @{
    "primary__compressed_plus_losses" = "Primary: compressed + losses"
    "primary__compressed_only" = "Primary: compressed only"
    "primary__all_compressed_features" = "Primary: all compressed features"
    "no_time_adv__compressed_plus_losses" = "No time adversary"
    "no_student__compressed_plus_losses" = "No student loss"
}

$palette = @{
    "CL" = "#2563eb"
    "C" = "#f97316"
    "ACF" = "#16a34a"
    "-T" = "#7c3aed"
    "-S" = "#dc2626"
}

function D($value) {
    return [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
}

function F($value, [string]$format = "0.00") {
    return (D $value).ToString($format, [Globalization.CultureInfo]::InvariantCulture)
}

function Pct($value) {
    return ((D $value) * 100.0).ToString("0", [Globalization.CultureInfo]::InvariantCulture)
}

function Esc($text) {
    return [System.Security.SecurityElement]::Escape([string]$text)
}

function ColorRamp([double]$value, [double]$min = 0.0, [double]$max = 1.0) {
    $t = [Math]::Max(0.0, [Math]::Min(1.0, ($value - $min) / ($max - $min)))
    $r0 = 248; $g0 = 250; $b0 = 252
    $r1 = 37;  $g1 = 99;  $b1 = 235
    $r = [int]($r0 + ($r1 - $r0) * $t)
    $g = [int]($g0 + ($g1 - $g0) * $t)
    $b = [int]($b0 + ($b1 - $b0) * $t)
    return ("#{0:x2}{1:x2}{2:x2}" -f $r, $g, $b)
}

function TextColor([double]$value, [double]$min = 0.0, [double]$max = 1.0) {
    $t = [Math]::Max(0.0, [Math]::Min(1.0, ($value - $min) / ($max - $min)))
    if ($t -ge 0.58) { return "#ffffff" }
    return "#111827"
}

function Write-Svg($path, [System.Collections.Generic.List[string]]$svg) {
    $svg.Add("</svg>")
    [System.IO.File]::WriteAllText($path, ($svg -join "`n"), [System.Text.UTF8Encoding]::new($false))
}

function Add-Key([System.Collections.Generic.List[string]]$svg, [int]$x, [int]$y) {
    $key = "CL main, C comp only, ACF all comp, -T no time, -S no student"
    $svg.Add("<text x=""$x"" y=""$y"" font-family=""Arial, sans-serif"" font-size=""7.2"" fill=""#4b5563"">$(Esc $key)</text>")
}

function New-HeatmapFigure {
    param(
        [string]$Title,
        [object[]]$Rows,
        [string[]]$Metrics,
        [hashtable]$MetricLabels,
        [string]$Path,
        [double]$Min = 0.0,
        [double]$Max = 1.0
    )

    $w = 345
    $h = 215
    $left = 42
    $top = 42
    $cellW = 68
    $cellH = 24
    $gap = 5
    $svg = [System.Collections.Generic.List[string]]::new()

    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""3.45in"" height=""2.15in"" viewBox=""0 0 $w $h"">")
    $svg.Add("<rect width=""$w"" height=""$h"" fill=""#ffffff""/>")
    $svg.Add("<text x=""10"" y=""16"" font-family=""Arial, sans-serif"" font-size=""11"" font-weight=""700"" fill=""#111827"">$(Esc $Title)</text>")
    Add-Key $svg 10 29

    for ($m = 0; $m -lt $Metrics.Count; $m++) {
        $x = $left + $m * $cellW
        $svg.Add("<text x=""$($x + $cellW / 2)"" y=""38"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7.5"" font-weight=""700"" fill=""#374151"">$(Esc $MetricLabels[$Metrics[$m]])</text>")
    }

    for ($r = 0; $r -lt $Rows.Count; $r++) {
        $row = $Rows[$r]
        $code = $shortLabels[$row.experiment]
        $y = $top + $r * ($cellH + $gap)
        $svg.Add("<text x=""34"" y=""$($y + 16)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""9"" font-weight=""700"" fill=""$($palette[$code])"">$(Esc $code)</text>")
        for ($m = 0; $m -lt $Metrics.Count; $m++) {
            $metric = $Metrics[$m]
            $value = D $row.$metric
            $x = $left + $m * $cellW
            $fill = ColorRamp $value $Min $Max
            $textFill = TextColor $value $Min $Max
            $svg.Add("<rect x=""$x"" y=""$y"" width=""$($cellW - 4)"" height=""$cellH"" rx=""2"" fill=""$fill"" stroke=""#e5e7eb"" stroke-width=""0.6""/>")
            $svg.Add("<text x=""$($x + ($cellW - 4) / 2)"" y=""$($y + 16)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""8.2"" font-weight=""700"" fill=""$textFill"">$(Pct $value)</text>")
        }
    }

    $legendY = $top + $Rows.Count * ($cellH + $gap) + 12
    $svg.Add("<text x=""$left"" y=""$legendY"" font-family=""Arial, sans-serif"" font-size=""7.2"" fill=""#6b7280"">Cell values are percentages; darker cells indicate higher metric values.</text>")
    Write-Svg $Path $svg
}

function New-TradeoffFigure {
    param(
        [object[]]$Rows,
        [string]$Path
    )

    $w = 345
    $h = 215
    $left = 44
    $top = 34
    $plotW = 270
    $plotH = 142
    $yMin = 0.80
    $yMax = 1.00
    $svg = [System.Collections.Generic.List[string]]::new()

    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""3.45in"" height=""2.15in"" viewBox=""0 0 $w $h"">")
    $svg.Add("<rect width=""$w"" height=""$h"" fill=""#ffffff""/>")
    $svg.Add("<text x=""10"" y=""16"" font-family=""Arial, sans-serif"" font-size=""11"" font-weight=""700"" fill=""#111827"">OOD recall-specificity tradeoff</text>")
    Add-Key $svg 10 29

    foreach ($tick in @(0.0, 0.5, 1.0)) {
        $x = $left + $tick * $plotW
        $svg.Add("<line x1=""$x"" y1=""$top"" x2=""$x"" y2=""$($top + $plotH)"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $svg.Add("<text x=""$x"" y=""$($top + $plotH + 15)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$tick</text>")
    }
    foreach ($tick in @(0.8, 0.9, 1.0)) {
        $y = $top + $plotH - (($tick - $yMin) / ($yMax - $yMin) * $plotH)
        $svg.Add("<line x1=""$left"" y1=""$y"" x2=""$($left + $plotW)"" y2=""$y"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $svg.Add("<text x=""$($left - 7)"" y=""$($y + 2.5)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$tick</text>")
    }
    $svg.Add("<line x1=""$left"" y1=""$top"" x2=""$left"" y2=""$($top + $plotH)"" stroke=""#6b7280"" stroke-width=""0.8""/>")
    $svg.Add("<line x1=""$left"" y1=""$($top + $plotH)"" x2=""$($left + $plotW)"" y2=""$($top + $plotH)"" stroke=""#6b7280"" stroke-width=""0.8""/>")

    foreach ($row in $Rows) {
        $code = $shortLabels[$row.experiment]
        $xVal = D $row.attack_recall
        $yVal = D $row.benign_specificity
        $x = $left + $xVal * $plotW
        $y = $top + $plotH - (($yVal - $yMin) / ($yMax - $yMin) * $plotH)
        $x = [Math]::Max($left, [Math]::Min($left + $plotW, $x))
        $y = [Math]::Max($top, [Math]::Min($top + $plotH, $y))
        $fill = $palette[$code]
        $labelX = $x + 6
        $labelY = $y - 5
        if ($code -eq "ACF") { $labelX = $x + 8; $labelY = $y + 13 }
        if ($code -eq "-S") { $labelX = $x + 8; $labelY = $y - 3 }
        if ($code -eq "-T") { $labelX = $x + 8; $labelY = $y + 8 }
        if ($code -eq "CL") { $labelX = $x - 18; $labelY = $y + 12 }
        $svg.Add("<circle cx=""$x"" cy=""$y"" r=""4.2"" fill=""$fill"" stroke=""#ffffff"" stroke-width=""1""/>")
        $svg.Add("<text x=""$labelX"" y=""$labelY"" font-family=""Arial, sans-serif"" font-size=""8"" font-weight=""700"" fill=""$fill"">$(Esc $code)</text>")
    }

    $svg.Add("<text x=""$($left + $plotW / 2)"" y=""$($h - 8)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""8"" fill=""#374151"">Attack recall</text>")
    $svg.Add("<text x=""12"" y=""$($top + 52)"" transform=""rotate(-90 12,$($top + 52))"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""8"" fill=""#374151"">Benign specificity</text>")
    Write-Svg $Path $svg
}

function New-DeltaFigure {
    param(
        [object[]]$Rows,
        [string]$Path
    )

    $baseline = $Rows | Where-Object { $_.experiment -eq "primary__compressed_plus_losses" } | Select-Object -First 1
    $others = $Rows | Where-Object { $_.experiment -ne "primary__compressed_plus_losses" }
    $w = 345
    $h = 215
    $left = 56
    $top = 40
    $axisX = 305
    $scale = 245
    $rowH = 31
    $svg = [System.Collections.Generic.List[string]]::new()

    $svg.Add("<svg xmlns=""http://www.w3.org/2000/svg"" width=""3.45in"" height=""2.15in"" viewBox=""0 0 $w $h"">")
    $svg.Add("<rect width=""$w"" height=""$h"" fill=""#ffffff""/>")
    $svg.Add("<text x=""10"" y=""16"" font-family=""Arial, sans-serif"" font-size=""11"" font-weight=""700"" fill=""#111827"">Drop from CL baseline</text>")
    $svg.Add("<text x=""10"" y=""29"" font-family=""Arial, sans-serif"" font-size=""7.2"" fill=""#4b5563"">Bars show metric delta vs. primary compressed + losses.</text>")

    foreach ($tick in @(-1.0, -0.5, 0.0)) {
        $x = $axisX + $tick * $scale
        $svg.Add("<line x1=""$x"" y1=""$top"" x2=""$x"" y2=""$($top + $rowH * $others.Count + 5)"" stroke=""#e5e7eb"" stroke-width=""0.7""/>")
        $svg.Add("<text x=""$x"" y=""$($top + $rowH * $others.Count + 18)"" text-anchor=""middle"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#6b7280"">$tick</text>")
    }
    $svg.Add("<line x1=""$axisX"" y1=""$top"" x2=""$axisX"" y2=""$($top + $rowH * $others.Count + 5)"" stroke=""#111827"" stroke-width=""0.8""/>")

    for ($i = 0; $i -lt $others.Count; $i++) {
        $row = $others[$i]
        $code = $shortLabels[$row.experiment]
        $y = $top + $i * $rowH
        $svg.Add("<text x=""45"" y=""$($y + 15)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""8.5"" font-weight=""700"" fill=""$($palette[$code])"">$(Esc $code)</text>")

        $deltaRecall = (D $row.attack_recall) - (D $baseline.attack_recall)
        $deltaF1 = (D $row.macro_f1) - (D $baseline.macro_f1)
        $pairs = @(
            @{ value = $deltaRecall; yoff = 4; color = "#2563eb"; label = "R" },
            @{ value = $deltaF1; yoff = 16; color = "#059669"; label = "F1" }
        )
        foreach ($pair in $pairs) {
            $x = $axisX + $pair.value * $scale
            $barX = [Math]::Min($x, $axisX)
            $barW = [Math]::Abs($axisX - $x)
            $svg.Add("<rect x=""$barX"" y=""$($y + $pair.yoff)"" width=""$barW"" height=""8"" rx=""1.5"" fill=""$($pair.color)""/>")
            $svg.Add("<text x=""$($barX - 4)"" y=""$($y + $pair.yoff + 7)"" text-anchor=""end"" font-family=""Arial, sans-serif"" font-size=""6.6"" fill=""#111827"">$($pair.value.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture))</text>")
        }
    }
    $legendY = 191
    $svg.Add("<rect x=""56"" y=""$legendY"" width=""8"" height=""8"" fill=""#2563eb""/><text x=""68"" y=""$($legendY + 7)"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#374151"">Attack recall</text>")
    $svg.Add("<rect x=""143"" y=""$legendY"" width=""8"" height=""8"" fill=""#059669""/><text x=""155"" y=""$($legendY + 7)"" font-family=""Arial, sans-serif"" font-size=""7"" fill=""#374151"">Macro F1</text>")
    Write-Svg $Path $svg
}

$orderedOod = foreach ($name in $order) { $ood | Where-Object { $_.experiment -eq $name } | Select-Object -First 1 }
$orderedMc = foreach ($name in $order) { $mc | Where-Object { $_.experiment -eq $name } | Select-Object -First 1 }

$summary = foreach ($row in $orderedOod) {
    $matchingMc = $orderedMc | Where-Object { $_.experiment -eq $row.experiment } | Select-Object -First 1
    $baseline = $orderedOod | Where-Object { $_.experiment -eq "primary__compressed_plus_losses" } | Select-Object -First 1
    [pscustomobject]@{
        experiment = $row.experiment
        code = $shortLabels[$row.experiment]
        display_name = $displayLabels[$row.experiment]
        ood_attack_recall = D $row.attack_recall
        ood_macro_f1 = D $row.macro_f1
        ood_bal_acc = D $row.bal_acc
        benign_specificity = D $row.benign_specificity
        unseen_recall = D $row.unseen_recall
        in_dist_hier_f1 = D $matchingMc.hier_f1
        in_dist_macro_f1 = D $matchingMc.macro_f1
        delta_attack_recall_vs_primary = (D $row.attack_recall) - (D $baseline.attack_recall)
        delta_macro_f1_vs_primary = (D $row.macro_f1) - (D $baseline.macro_f1)
    }
}
$summary | Export-Csv (Join-Path $OutputDir "ablation_summary.csv") -NoTypeInformation

New-HeatmapFigure `
    -Title "Final OOD metrics" `
    -Rows $orderedOod `
    -Metrics @("attack_recall", "unseen_recall", "macro_f1", "benign_specificity") `
    -MetricLabels @{
        "attack_recall" = "Atk R"
        "unseen_recall" = "Unseen"
        "macro_f1" = "F1"
        "benign_specificity" = "Spec"
    } `
    -Path (Join-Path $OutputDir "ood_ablation_metrics.svg")

New-HeatmapFigure `
    -Title "In-distribution metrics" `
    -Rows $orderedMc `
    -Metrics @("bal_acc", "macro_f1", "hier_f1", "subtype_f1") `
    -MetricLabels @{
        "bal_acc" = "Bal"
        "macro_f1" = "F1"
        "hier_f1" = "Hier"
        "subtype_f1" = "Sub"
    } `
    -Path (Join-Path $OutputDir "multiclass_ablation_metrics.svg") `
    -Min 0.75

New-TradeoffFigure -Rows $orderedOod -Path (Join-Path $OutputDir "ood_precision_recall.svg")
New-DeltaFigure -Rows $orderedOod -Path (Join-Path $OutputDir "ablation_delta_vs_primary.svg")

Write-Host "wrote compact two-column SVG figures to $OutputDir"
