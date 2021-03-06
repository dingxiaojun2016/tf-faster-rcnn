# 非极大抑制（NMS,Non-maximum suppression）
## NMS算法介绍
RCNN会从一张图片中找出n个可能是物体的矩形框，然后为每个矩形框为做类别分类概率，定位一个车辆，
最后算法就找出了一堆的方框，我们需要判别哪些矩形框是没用的。非极大值抑制的方法是：先假设有6个
矩形框，根据分类器的类别分类概率做排序，假设从小到大属于车辆的概率 分别为A、B、C、D、E、F。<br>
* 从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;<br>
* 假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。<br>
* 从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，
那么就扔掉；并标记E是我们保留下来的第二个矩形框。<br>

就这样一直重复，找到所有被保留下来的矩形框。<br>
非极大值抑制（NMS）顾名思义就是抑制不是极大值的元素，搜索局部的极大值。这个局部代表的是一个邻
域，邻域有两个参数可变，一是邻域的维数，二是邻域的大小。这里不讨论通用的NMS算法，而是用于在目
标检测中用于提取分数最高的窗口的。例如在行人检测中，滑动窗口经提取特征，经分类器分类识别后，
每个窗口都会得到一个分数。但是滑动窗口会导致很多窗口与其他窗口存在包含或者大部分交叉的情况。
这时就需要用到NMS来选取那些邻域里分数最高（是行人的概率最大），并且抑制那些分数低的窗口。
## 目录架构
* py_cpu_nms.py<br>
    --通过python直接编写的nms算法，跑在cpu上，效率较低。
* cpu_nms.pyx<br>
    --通过cython编写的nms算法，跑在cpu上，因为会转换成c库,所以工作起来要比纯python写的nms
    算法效率要高很多。
* gpu_nms.pyx+nms_kernel.cu+gpu_nms.hpp<br>
    --通过cython和cuda编写nms算法，使该算法跑在gpu上，这样计算会比cpu版cython编写的算法要
    更高效很多。
* 其他c文件和so库均为cython转换自动生成的，不需要关心。
## 编译
* 先安装cuda+cudnn环境，如下是我个人安装过程和遇到的坑，不同环境遇到的问题会不同，确保环境完全安装好并检查通过。<br>
    https://blog.csdn.net/zcc450959507/article/details/89672332
* 修改setup.py脚本中gpu -arch参数，参考如下：(nvdia官网如下链接能找到对应的架构https://developer.nvidia.com/cuda-gpus )<br>

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | GTX 1070 | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

* 根据setup.py代码，需要设置一下cuda home环境变量：<br>
    export CUDAHOME="/usr/local/cuda"<br>
    在lib根目录，执行make clean && make all，就会生成cython文件对应的c源文件和运行so库。
## 代码分析
* 目前分析了python直接编写的nms算法和通过cython编写的nms算法，具体过程见源代码文件。
## 实际测试
使用不同版本的nms运行demo.py程序对demo图片进行目标检测，运行速度结果如下：<br>
gpu-nms:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/000542.jpg
Detection took 0.093s for 300 object proposals

Demo for data/demo/001150.jpg
Detection took 0.099s for 300 object proposals

Demo for data/demo/001763.jpg
Detection took 0.089s for 300 object proposals

Demo for data/demo/004545.jpg
Detection took 0.091s for 300 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cpu-nms:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/000542.jpg
Detection took 2.466s for 300 object proposals

Demo for data/demo/001150.jpg
Detection took 2.452s for 300 object proposals

Demo for data/demo/001763.jpg
Detection took 2.464s for 300 object proposals

Demo for data/demo/004545.jpg
Detection took 2.470s for 300 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pure-python-nms:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/000542.jpg
Detection took 2.443s for 300 object proposals

Demo for data/demo/001150.jpg
Detection took 2.438s for 300 object proposals

Demo for data/demo/001763.jpg
Detection took 2.448s for 300 object proposals

Demo for data/demo/004545.jpg
Detection took 2.467s for 300 object proposal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
