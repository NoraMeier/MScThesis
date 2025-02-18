# MScThesis

You will find the videos based on the uncertainty predictions, as well as the videos with the enhanced frame rate in the EMA-VFI/final_videos folder.

The density plots, and all the interpolated and base frames can be found via the EMA-VFI/video folder, and are separated y original video.

There are also a number of scripts:
  - break_video.py will take a video in one of the video subfolders and break it into frames for examination
  - demo_2x.py performs the interpolation. Leave the model argument blank, as no other model architectures were trained. input_dir is relevant if you wish to perform one interpolation, in which case this argument should be the path to a folder containing two input images named img1.jpg and img3.jpg. the uncertainty argument specifies which type of uncertainty quantification should be performed, 'none' will use the base model. Finally, the video argument specifies which video should be interpolated, if not specified, it will perform one interpolation on the images specified by the inpput_dir argument.
  - enhance_framerate.py enhances the framerate of an existing video. Note that it does not create the new video itself, only saves all the frames to a folder.
  - make_densityplot_video.py generates the density plots and correlation curves for a video
  - train.py trains a new model
