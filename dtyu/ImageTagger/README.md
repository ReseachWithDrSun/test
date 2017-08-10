# ImageTagger: X-ray Scattering Image Tagger with Deep Learning Prediction Assist

ImageTagger is configured to run on node04 of HPC1 cluster.

## Setup

1. Log in node04 with X11 forwarding (`-X`). Mac users need to install XQuartz.

2. Add the following lines to your `~/.bash_rc` file on node04:

	# anaconda, tensorflow and cuda support
	export LD_LIBRARY_PATH=/software/cuda/7.5/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
	export CUDA_HOME=/software/cuda/7.5
	export PATH="/software/anaconda3/bin:$PATH"

3. Relog in and do a test run:

	python /lscr/home/zquan/ImageTagger/main.py

4. Get a latest copy of the program via `git clone` and upload it to your home
directory and run. Write access is required to modify/save configurations so the
program needs to be in your directory. Also make sure **you have write access to
the dataset to tag**.
