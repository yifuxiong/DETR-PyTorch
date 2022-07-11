## DeTR-Lite

A simple version of DeTR.



### Before you enjoy this DeTR-Lite

---

The purpose of this project is to allow you to learn the basic knowledge of DeTR. 
Perfectly reproducing DeTR is not my goal because of the high demand for computing resources.

The input size used by the official DeTR is 800x1333, but my device is 
not enough to support such a large input size, so I adjusted it to 640x640.

The project can now train and eval my DeTR-Lite, 
but it is uncertain whether the desired effect can be achieved, 
and whether there are potential bugs has not yet been discovered.

The writing of the project code is in progress ...



### Flow

---



#### 1. backbone

---

（1）resnet50为例，输入图像shape为(B, 3, H, W) -> (B, 2048, H0, W0)，下采样32倍，即H0 = 1/32 H, W0 = 1/32 W。

（2）再经过一个input_proj，即几个1x1的Conv，将通道压缩为256，shape变为(B, 256, H0, W0)。

（3）为了适应Transformer的输入，将shape从(B, 256, H0, W0) -> (B, H0 x W0, 256) -> (B, N, 256)，其中 N = H0 x W0。



并且，输入Transformer时要带Pos Encoding一起，之前已经讨论过Pos Embedding和Pos Encoding的区别了，这里不再复述。只说一下输入的Pos Encoding的shape。

Pos Encoding的维度得匹配的上Patch Embedding，因此shape = (1, N, 256)，N = H0 x W0。这里的B=1，在运行时自动调用python的广播机制，对每个B维度上都填充相同的Patch Embedding。



---

《Attention is All You Need》中的pos encoding的三角函数计算公式：
$$
PE_{(pos, 2i)} = sin(\frac{pos}{10000^{2i / d_{model}}}) \\
PE_{(pos, 2i + 1)} = cos(\frac{pos}{10000^{2i / d_{model}}})
$$

---



为了保证二维的特性，需要为在X和Y两个维度都去计算Position Embedding。相应的代码如下：

```python
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
```

在forward()中，`x = tensor_list.tensors`，这个就是特征图P5，shape=(B, N, 256)。

*注意：通过resnet50后的 feature map 是C5，而 C5 再通过 input_proj 得到的 feature map 就是 P5*



代码中的`not_mask = ~mask`。是DETR代码中采用了多尺度增强，对两张尺寸不同的图片贴上mask补零成一样size的feature map。

DeTR的做法是采用padding补零的方式，相关代码如下：mask就是从这里生成的。它的作用就是记录训练时的图像哪一部分是有效的，哪一部分是无效的。显然，无效的就是指被padding上的那一块。

```python
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)
```



#### 2. Transformer

---

Transformer部分分为Encoder和Decoder，下面跟着图一起走，思路会很清晰。

![](https://pic4.zhimg.com/80/v2-5539f2d7875adfa237a9c504d182f25b_720w.jpg)

##### Encoder

---

Encoder部分是经典ViT中的Encoder，注意循环次数为N（左上角有个xN）。



##### Decoder

---

​		从Encoder进来的Embedding在Decoder中叫做memory，再新输入一个向量object_queries。tgt由上一次的tgt和object queries作self-attention得来，最初的tgt默认为0向量。看代码跟着图走，思路很清晰。



---

*_key_padding_mask和attn_mask的区别*

（1）key_padding_mask指的是在encoder和Decoder的输入中，由于每个batch的序列长短不一，被padding的内容需要用key_padding_mask来标识出来，然后在计算注意力权重的时候忽略掉这部分信息。

同时，尤其值得一提的是谷歌开源的bert代码中，里面的参数attention_mask其实指的就是这里的key_padding_mask.

举个例子，3个batch的长短不一，这里用pad填充：
$$
(a, b, c, pad) \\
(a, b, c, d) \\
(a, b, pad, pad)
$$
对应的mask就为:
$$
(True, True, True, False) \\
(True, True, True, True) \\
(True, True, False, False)
$$
（2）而attn_mask只用于Decoder训练时的解码过程，是类似于NLP中的一种计算方式，掩盖掉当前时刻之后的信息，只能读取上文信息，比如一个句子$(a, b, c, d)$。

读到a时刻时，句子为(pad, pad, pad, pad)，只能读到空；

读到b时刻时，句子为(a, pad, pad, pad)，只能读到a；

读到c时刻时，句子为(a, b, pad, pad)，能读到a, b；

...

---



<img src="https://pic4.zhimg.com/80/v2-5539f2d7875adfa237a9c504d182f25b_720w.jpg" style="zoom: 67%;" />



注意，一层decoder里面包含两个Attention模块，分别是self-attention和cross-attention。



**self-attention**

​		object queries作为decoder的输入，并从其中得到Q, K和V，它的shape=[Nq,  256]，这里的shape其实是[H0*W0, C]，是一个可学习的向量，其中Nq是输出目标个数（DETR中Nq默认为100），C是维度（通道数）。然后在batchsize维度上进行广播，shape=[B, Nq, C]。



对于第一个attention，Qp为Object Queries，$Q_1, K_1$表示encoder总共有M层，其中第1层的Q和K。每个变量计算如下：
$$
Q1 = K1 = tgt + Qp \\
V1 = tgt
$$
其中tgt是decoder第1层的输入（由object queries自己跟自己做self-attention计算得到），通常会被初始化0。标准的self-attention计算如下，得到输出的tgt：
$$
tgt = softmax(\frac{Q_1 K_1^T}{\sqrt{d_k}}) V_1
$$


**cross-attention**

​		现在进入第二个attention模块。第二个是交叉attention，即Q, K, V不全是由输入得来的。具体对于Q, K, V来说，分别有：

Q2来自第一个self-attention的输出和object queries，两者的和。（注意这里的tgt已经是通过了self-attention之后的结果）
$$
Q2 = tgt + Qp
$$
而K2和V2来自于Encoder的输出，代码中记作memory：
$$
K2 = memory + pos \\
V2 = memory
$$
​		同样，这里的V还是不加position embedding（原版的Transformer的V是加入了position embedding）。继续进行attention计算，只不过这里不是自己和自己做attention了，所以不叫self-attention，而是叫cross-attention。
$$
tgt2 = softmax(\frac{Q_2 K_2^T}{\sqrt{d_k}}) V_2
$$


附：第2个cross-attention得到的tgt2，代码中如下：

```python
# 注意这里，参考图中value加入了position embedding，而代码这里没有加上去
# 可以尝试自己加上去，跑一边对比结果
tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
```



​		由此，最初的tgt经过两个attention模块（self-attention和cross-attention）的一层decoder的处理后，得到了一次更新。总结一下，完整的decoder中的所有变量更新公式如下：

第一层`self-attention`，其中`tgt`初始为0向量，`Qp`是Object Queries。
$$
Q_1 = K_1 = tgt + Q_p \\
V_1 = tgt \\
tgt = softmax(\frac{Q_1 K_1^T}{\sqrt{d_k}}) V_1
$$
第二层`cross-attention`，Encoder输出的embedding记为`memory`。
$$
Q_2 = tgt + Q_p \\
K_2 = memory + pos \\
V_2 = memory \\
tgt2 = softmax(\frac{Q_2 k_2^T}{\sqrt{d_k}}) V_2
$$
​		再补充一点，Decoder的输出包含M层（源代码中M默认为6），每一层输出的tgt都可以拿出来，tgt.shape=[B, Nq, C=256]。

​		如果我们不要中间输出，那就只保留decoder最后一层的输出。由于detr在后面计算loss的时候，会考虑对M进行遍历，毕竟中间输出也的确可以作为比较，所以为了格式的统一，我们给它多加上一个M的维度，即[B, Nq, C] -> [M, B, Nq, C]，其中M默认为1。



#### 3. post work

---

decoder最终的输出，要对它进行分类和回归预测，因此分别进入两个FFN（也叫MLP），代码分别如下：

```python
# 分类
self.cls_det = nn.Linear(args.hidden_dim, num_classes + 1)
# 回归
self.reg_det = MLP(args.hidden_dim, args.hidden_dim, 4, 3)
```



```python
# transformer
h = self.transformer(x, self.query_embed.weight, self.pos_embed)[0]

# output: [M, B, N, C] where M = num_decoder since we use all intermediate outputs of decoder
outputs_class = self.cls_det(h)
outputs_coord = self.reg_det(h).sigmoid()
```



代码中，h则是transformer的输出，即最终的tgt，shape=[M, B, Nq, C=256]。

outputs_classes.shape=[M, B, Nq, K+1]，K是目标类别数，加上1表示背景；

outputs_coord.shape=[M, B, Nq, 4]，分别是bbox的中心点坐标和宽高，都是相对值，所以后面用了sigmoid。



*题外话*

​		作回归预测的时候，DETR的这一点和YOLO、RetinaNet是很不同的，直接回归坐标，没有显式的anchor概念。尽管有些研究学者认为object queries是一种anchor，但这种anchor也已经和以前用的anchor概念截然不同了。但这有一个潜在的问题就是直接回归边界框坐标，没有anchor的概念，那么怎么能确保训练的稳定性呢？换言之，Transformer输出的tgt.shape=[B, M, Nq, C=256]中的Nq是预测的物体个数，那序列的每个元素负责哪里的预测呢？

​		在这一点上，object queries是没有显示的位置概念在里面的，不像CNN中的那种天然的anchor-based概念来得直观且清晰，这同时也是DETR需要耗费大量的训练时间的原因之一。最近，旷视提出了的Anchor DETR的工作，给object queries赋予了位置先验，来尝试解决这一问题。



Decoder输出后留下来了两个问题：

1. 什么是object queries？它是干嘛的？

2. 没有了类似于CNN中的anchor的概念，那么该如何寻找正样本并计算损失呢？



#### 4. Object Queries

---

我的思路：Decoder的输入Object Queries，可以替换为从另一个轻量级CNN提取出来的特征，然后展开成Embedding。

先看看现有文献都怎么说：





#### 5. Hungarian Match

---













### Environment Config

---

在配置detectron2环境的时候，必须经过这一步。首先需要安装visual studio 2015~2019，然后将cl.exe配置到高级变量中。



百度：windows上安装detectron2出问题 https://blog.51cto.com/u_14558366/5179129

visual studio 2019企业版激活 https://www.bilibili.com/video/av503650755

百度：cl.exe https://blog.csdn.net/qq_42868008/article/details/81987852

如果是cuda10.2，那么visual studio的版本必须是2015到2019版本之间。



detectron2

```text
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```



pytorch

```text
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.2 -c pytorch
```



Install fvcore

```text
pip install fvcore==0.1.1.post20200716
```



pycocotools，官方包给的只能运行在Linux下，Windows有大佬修改出来了

链接https://github.com/philferriere/cocoapi

或者https://gitcode.net/mirrors/philferriere/cocoapi?utm_source=csdn_github_accelerator

```text
cd cocoapi/PythonAPI/
python setup.py install
```



在Windows下安装，可以参考“太阳花的小绿豆”，这个做法最简单。

```
pip install pycocotools-windows
```



panopticapi，做全景分割任务。gcc的版本一般是7.1以上，我的是8.1.0

```text
git clone https://github.com/cocodataset/panopticapi.git
cd panopticapi
pip install -e .
```

...



### References

---

[DeTR-Lite   zhihu](https://zhuanlan.zhihu.com/p/421170951)