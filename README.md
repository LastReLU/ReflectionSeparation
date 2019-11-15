# Reflection Separation Sirius project

## How to run
0.  apt-get install libglib2.0-0 ;
    apt-get install -y libsm6 libxext6 libxrender-dev;
    pip install opencv-python
1. Download data:
    ```bash
    wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
    python download_data.py
    ```
2. Run training:
    ```bash
        python train.py --batch_size=64 --logs ./runs/001
    ```

3. Watch the logs
    ```bash
        CUDA_VISIBLE_DEVICES= tensorboard --logdir=./runs
    ```

## Dataloader

From every image let's generate following entry:
```
dict(
    image=numpy fp32 array (128, 128, 3) with numbers (-1, 1),
    reflections=[(alpha, double_reflected_blurred_image), ...]
)
```


## Papers:
1. Single Image Reflection Removal Using Deep Encoder-Decoder Network. https://arxiv.org/pdf/1802.00094.pdf

## Data
1. Outdoor. https://www.hel-looks.com/about/
2. Indoor. http://web.mit.edu/torralba/www/indoor.html

## People
1. Alexey Ozerin
2. Sergey Kim
3. Andrey Bocharnikov
4. Ivan Lazunin
