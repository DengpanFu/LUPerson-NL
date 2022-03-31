# The Process to build LUPerson-NL

All the videos' YouTube key can be found at [vnames.txt](https://drive.google.com/file/d/19XPcO61QGrcNcqYE1pRZ1Hk3Z4CNu-6X/view?usp=sharing)

All detection results can be found at [dets.zip](https://drive.google.com/file/d/1-bFKMRglmD_e3wdtq4jlpjX3N3xGo2d2/view?usp=sharing)

**!! The following scripts are not well tested, you may fail to download some videos, but the scripts provide the main procedure !!**.

## Download the raw videos
```
python download.py -f ${YOUR_VIDEO_NAME_FILE_DIR}/vname.txt -s ${YOUR_VIDEO_DIR}
```
[youtube-dl](https://github.com/ytdl-org/youtube-dl) is needed.

## Extract images from raw videos and their detections
```
python extract.py -v ${YOUR_VIDEO_DIR} -d ${DETECTION_DIR} -s ${SAVE_DIR}
```

## Convert extracted images to lmdb data
```
python convert_lmdb.py
```

# You can also download it from [BaiDuDisk](https://pan.baidu.com/s/1sNV62vxm2VtgkVa7V4VmIA) code:plnl
