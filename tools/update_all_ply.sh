#!/bin/bash
set -e

DIR1="nerf_160826_0401_8-15_volume_three"
DIR2="nerf_160826_0508_8-15_volume_one"

for DIR in "$DIR1" "$DIR2"; do
    for TYPE in "obb" "full"; do
        INPUT="${DIR}/pointcloud_raw_${TYPE}.ply"
        SOLID="${DIR}/pointcloud_raw_${TYPE}_solid_table.ply"
        GIF="${DIR}/pointcloud_raw_${TYPE}_solid_table.gif"
        
        # Clean up the old bad reprojected files to avoid confusion
        rm -f "${DIR}/pointcloud_raw_${TYPE}_solid_table_reprojected.ply"
        rm -f "${DIR}/pointcloud_raw_${TYPE}_solid_table_reprojected.gif"
        
        if [ -f "$INPUT" ]; then
            echo "Processing $INPUT..."
            .venv_nerf/bin/python tools/fill_25d_extrusion.py --input "$INPUT" --output "$SOLID"
            .venv_nerf/bin/python tools/export_bio_format.py --ply "$SOLID"
            .venv_nerf/bin/python tools/render_pointcloud_video.py --ply "$SOLID" --out "$GIF"
        fi
    done
done
echo "All done!"
