# Path to the output file
param (
    [string]$outputFile = "FolderStructure.txt",
    [int]$fileThreshold = 20,
    [string[]]$excludePatterns = @(".git", "node_modules", "__pycache__"),
    [switch]$showSizes,
    [switch]$showHidden
)

# Initialize the output content
$output = [System.Collections.Generic.List[string]]::new()

# Add file/folder size calculation
function Get-FriendlySize {
    param ([long]$bytes)
    $sizes = 'B','KB','MB','GB','TB'
    $order = 0
    while ($bytes -ge 1024 -and $order -lt $sizes.Length) {
        $bytes /= 1024
        $order++
    }
    return "{0:N2} {1}" -f $bytes, $sizes[$order]
}

# Add time formatting function after Get-FriendlySize
function Get-FriendlyTimeSpan {
    param ([System.TimeSpan]$timeSpan)
    if ($timeSpan.TotalMinutes -lt 60) {
        return "$([math]::Round($timeSpan.TotalMinutes, 1)) minutes ago"
    } elseif ($timeSpan.TotalHours -lt 24) {
        return "$([math]::Round($timeSpan.TotalHours, 1)) hours ago"
    } else {
        return "$([math]::Round($timeSpan.TotalDays, 1)) days ago"
    }
}

# Function to recursively process the folder structure
function Get-FolderStructure {
    param (
        [string]$folderPath,   # The path of the current folder
        [int]$indentLevel      # Indentation level for readability
    )

    # Generate indentation
    $indent = " " * $indentLevel

    # Add the current folder to the output
    $output.Add("$indent- $(Split-Path $folderPath -Leaf)")

    # Get all files in the current folder
    $files = Get-ChildItem -Path $folderPath -File -ErrorAction SilentlyContinue | Where-Object {
        $shouldInclude = $true
        foreach ($pattern in $excludePatterns) {
            if ($_.FullName -like "*$pattern*") {
                $shouldInclude = $false
                break
            }
        }
        $shouldInclude -and ($showHidden -or !$_.Attributes.HasFlag([System.IO.FileAttributes]::Hidden))
    }

    if ($files.Count -gt $fileThreshold) {
        # Summarize file types if there are more than $fileThreshold files
        $output.Add("$indent  More than $fileThreshold files, summary:")

        # Count files by extension
        $fileTypeCounts = @{}
        foreach ($file in $files) {
            $ext = $file.Extension.ToLower()
            if (-not $fileTypeCounts.ContainsKey($ext)) {
                $fileTypeCounts[$ext] = 0
            }
            $fileTypeCounts[$ext]++
        }

        # Add summary to the output
        foreach ($ext in $fileTypeCounts.Keys) {
            $output.Add("$indent    ${ext}: $($fileTypeCounts[$ext]) files")
        }
    } else {
        # List all files if there are $fileThreshold or fewer, including last modified time
        foreach ($file in $files) {
            $timeSpan = New-TimeSpan -Start $file.LastWriteTime -End (Get-Date)
            $sizeInfo = if ($showSizes) { " ($(Get-FriendlySize $file.Length))" } else { "" }
            $output.Add("$indent  $($file.Name) (modified $(Get-FriendlyTimeSpan $timeSpan))$sizeInfo")
        }
    }

    # Process subdirectories
    $subdirs = Get-ChildItem -Path $folderPath -Directory -ErrorAction SilentlyContinue
    foreach ($subdir in $subdirs) {
        Get-FolderStructure -folderPath $subdir.FullName -indentLevel ($indentLevel + 2)
    }
}

# Add progress indication
$progressPreference = 'Continue'
$filesProcessed = 0
$totalFiles = (Get-ChildItem -Recurse -File).Count

# Start processing from the current directory
Get-FolderStructure -folderPath (Get-Location).Path -indentLevel 0

# Save the output to a file
$output -join "`n" | Set-Content -Path $outputFile -Encoding UTF8

# Inform the user
Write-Output "Folder structure has been saved to '$outputFile'"

# Add summary statistics
$totalStats = @{
    TotalFiles = 0
    TotalSize = 0
    FileTypes = @{}
}

$output.Add("`nSummary:")
$output.Add("Total Files: $($totalStats.TotalFiles)")
$output.Add("Total Size: $(Get-FriendlySize $totalStats.TotalSize)")
$output.Add("File Types Distribution:")
foreach ($type in $totalStats.FileTypes.Keys | Sort-Object) {
    $output.Add("  $type : $($totalStats.FileTypes[$type])")
}
