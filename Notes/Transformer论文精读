# 1 Introduction
目前最新的方法是RNN，包括LSTM、GRU等等，主要是语言模型和encoder-decoder两类。

RNN的特点是：
给定一个序列，从左往右一步步进行计算，比如第t个词的隐藏状态$h_t$是由前一个隐藏状态$h_{t-1}$ 和当前的词$x_t$计算得出来的。
会存在两个缺陷: RNN是一个时序，很难进行并行计算,计算性能比较差；另一个方面，由于是一步一步往后进行计算的，如果时序很长的话，很早前的信息内容可能会丢失掉。

attention 已经被用于encoder-decoder了，主要是用到怎么把encoder的信息传递到decoder，是跟RNN一起使用的。

transformer是全新的模型，不再使用RNN，是完全基于attention mechanism的模型。可以进行并行计算。

# 2 Background
如何使用cnn替换rnn，但是cnn对长的时序难以建模。而transformer可以直接看到所有的像素。cnn好的点在于有多输出通道，transformer借鉴这种思想，使用了multi-head attention。

self-attention已经有很多前人做了，不是本文的创新。

# 3 Model Architecture
过去时刻的输出也是当前时刻的输入，叫自回归。
## 3.1 Encoder-Decoder 块
**Encoder**
encoder有6个块，每个块有两个层，一层是self-attention + 残差 + layernorm，还有一层是mlp+ 残差 + layernorm。设输入encoder的embedding是X(由word embedding + Positional embedding)
第一层：layernorm(X + attention(X)) = Y
第二层：layernorm(Y + FFN(Y)) 

**Decoder**
encoder有6个块，每个块有三个层，和encoder 一样，一层是self-attention + 残差 + layernorm，还有一层是mlp+ 残差 + layernorm，但是在前面有个带掩码的注意力机制。
同理：设输入encoder的embedding是X(由word embedding + Positional embedding)
第一层：layernorm(X + masked_attention(X)) = Y
第二层：layernorm(Y + attention(Y)) = Z 
第三层：layernorm(Z + FFN(Z)) 
![transformer架构图](https://upload-images.jianshu.io/upload_images/25141709-0e08580fc58c5d90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 3.2 Attention
Attention function 可以被描述为将一个query 和一个键值对的集合映射为一个output的一个函数，这里所有的q、k、v和输出都是向量。output是通过values的加权和计算出来的。

**因为同样一个矩阵向量，复制成了q、k、v三份，都是自身，所以叫自注意力机制。**
简单理解是Q = K = V，但是实际上$Q = XW_q$, $K = XW_k$, $V = XW_v$， 这就是self-attention 和attention的区别。
**从transformer的架构可以看出，有两个self-attention 和一个非self-attention，在encoder输出给到decoder部分的计算是encoder-decoder attention，是用来学习源句与目标句之间的关系**

![image.png](https://upload-images.jianshu.io/upload_images/25141709-edd0ff8655d3efb6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/25141709-3172b709041098b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 3.2.1 scaled dot-product attention
![scaled dot-product attention.png](https://upload-images.jianshu.io/upload_images/25141709-5ee41c39f076c7b1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$$
Q K^\mathrm T =>  \frac {Q K^\mathrm T}{\sqrt{d_k}} => Softmax (\frac {Q K^\mathrm T}{\sqrt{d_k}}) 
$$
$$
Attention(Q, K, V) = Softmax (\frac {Q K^\mathrm T}{\sqrt{d_k}})V
$$
**为什么要除以$\sqrt{d_k}$？**
当dk不是很大的时候，除不除以都没有关系，但是如果dk很大的时候=>点积就会变得比较大或者比较小=>值之间的相对差距变得更大=>最大的值softmax出来更加趋近于1，剩下的值更加趋近0，值会更加向两端靠拢=>计算梯度的时候，梯度会更小，跑不动。

mask是在训练的时候使得，t时刻的q只能看到v1.....vt-1，而不能看到后面的内容，更加的符合实际的思维。

### 3.2.2 multi-head attention

![image.png](https://upload-images.jianshu.io/upload_images/25141709-f91361fe6ff51974.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
multi-head 的详解见本人的文章*llms-from-scratch--attention mechanism 详解代码计算*

### 3.2.3 Applications of Attention in our Model
应用在transformer的attention主要有三个，（1）encoder-decoder attention（非self-attention）；（2）单纯的self-attention；（3）带有mask的self-attention
- encoder-decoder attention（非self-attention）：就是encoder的输出的K和V是decoder的输入，同时Q是来自decoder，这允许解码器中的每个位置都关注输入序列中的所有位置
- 单纯的self-attention：计算inputs的Q、K 和V，所有的Q、K、V来自相同的位置
- 带有mask的self-attention：decoder的输入，在计算权重的时候，不应该算上未来的token的权重，需要进行mask。解码器中的自注意力层允许解码器中的每个位置关注解码器中的所有位置，直到该位置并包括该位置。我们需要防止解码器中信息向左流动，以保留自回归属性。我们通过屏蔽（设置为 -∞）softmax 输入中对应于非法连接的所有值来实现这一点。
## 3.3 FFN-前馈神经网络
从transformer架构图中可以看到， 除了attention层外，每个编码器和解码器里都有一个全连接层，这包括两个线性变换和relu激活函数。
$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 
$$
输入和输出的维数为 dmodel = 512，内层的维数为 df f = 2048.
即$xW_1 + b_1$ 维度是2048， $max(0, xW_1 + b_1)W_2 + b_2 $又变成512维了。
## 3.4 embedding and softmax
encoder 的输入和output 的输入 以及在softmax前面的线性层也需要embedding，三个embedding 是一样的权重。并且把权重乘了$\sqrt{d_{model}}$，也就是512.

**Q：为什么需要乘以$\sqrt{d_{model}}$？**
从参考的各种资料来看，说embeding 的矩阵初始化是采用了Xavier初始化，正态总体Embedding matrix 的抽样分布满足：$N(0, 1/{\sqrt{d_{model}}})$,这会导致一个问题，Embedding matrix 元素分布的方差会随着变化，如果$d_{model}$较大，输出值的波动会比较小。通过乘以$\sqrt{d_{model}}$，可以使embedding matrix的分布回调到$N(0, 1)$ 
根据李沐大神的说法的自我理解：在学embedding向量的时候，会把每个向量的l2norm学的相对较小，不管维度多大，加和的值等于1，那么维度越大的时候，每个值的权重越小。但是需要加上后续的psitional encoding，这个不会把l2norm固定住，现在把word embedding 乘上$\sqrt{d_{model}}$， 就可以使两者在一个scale上进行相加了。
 ## 3.5 Positional Encoding
由于transformer 既没有RNN也没有CNN这种天然带有时序信息的网络，attention本身不具有时序信息，也就是说在计算q和key之间的距离的时候，前面的词怎么打乱，都不会影响这个结果，也就是说不会带有时序信息。在word embedding上需要加上相对应词的位置编码信息。这些信息可以加绝对的位置的 也可以加相对位置的。transformer使用了sine 和cosine的方法来表示，也就是Sinusoidal Positional Encoding， 这种方式是**绝对位置编码方式**，为序列中的每个位置（计算一个独特的、固定的向量。这个向量使用不同频率的正弦和余弦函数生成：
$$
PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}})  
$$
$$
PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})
$$
 
其中 pos 是位置，i 是维度。
$d_{model}$：输出嵌入空间的维度  【Transformer中为512】
pos：输入序列中的单词位置，0≤pos≤L-1 【序列中某个词的位置，位置从0开始，到长度-1】
i：用于映射到列索引, 其中0≤i<d/2，并且i 的单个值还会映射到正弦和余弦函数【i的取值从0开始，到512/2】
也就是说，位置编码的每个维度都对应于一个正弦曲线。由于频率是在1 - 1/10000之间，所以波长形成从 2π 到 10000 ·2π. 我们选择这个函数是因为我们假设它将允许模型轻松学习相对位置，因为对于任何固定偏移量 k，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。

**虽然Sinusoidal Positional Encoding有助于学习相对关系，但它本质上代表的是绝对位置。它不像真正的相对位置编码那样仅仅关注标记之间的距离。**

我们还尝试使用Learned Positional Encoding，发现这两个版本产生了几乎相同的结果，该方式跟本文一样都是绝对位置编码方法。
**Sinusoidal Positional Encoding 和 Learned Positional Encoding都是绝对位置编码，但是Sinusoidal Positional Encoding更具有的优势在于可能允许模型外推到比训练期间遇到的序列长度更长的序列长度**。
![图](https://upload-images.jianshu.io/upload_images/25141709-2432c78ed90712d1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


为什么$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数？
![](https://upload-images.jianshu.io/upload_images/25141709-2b345be89eef52f1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/25141709-65f38bcb7b16c708.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



Sinusoidal Positional Encoding的优点在于具有长度外推和隐式相对位置理解等优势，但它也存在局限性。
缺点在于，**SPE 的固定性可能导致其对未知序列长度的泛化能力较差，并且它可能无法捕捉到所有相对位置信息的细微差别**。
以下是更详细的分析：
1. 对未知序列长度的泛化能力有限：
SPE 基于固定的数学公式计算，因此可以外推到比训练期间看到的序列更长的序列。
然而，由于注意力模式泛化能力不佳等其他因素，模型在更长的序列上的整体性能可能仍然会下降。
与learned embedding不同，SPE 不需要针对新的序列长度进行重新训练，但模型解读编码信息的能力可能仍然有限。
2. 特定位置过拟合的可能性：
尽管 SPE 为每个位置提供了唯一的编码，但它仍然可能导致模型对序列中特定的标记位置过拟合。
发生这种情况的原因是位置信息被编码为固定模式，而模型可能会过度依赖这些模式，而不是理解 token 之间的潜在关系。
3. 隐式与显式相对位置表示：
SPE 以绝对方式编码位置信息，这意味着它编码的是 token 相对于序列开头的位置。
虽然正弦函数的固定特性允许模型隐式学习一些相对位置关系，但它可能不如专门设计用于直接表示相对位置的方法有效。
诸如旋转位置嵌入 (RoPE) 之类的方法旨在通过显式编码相对位置信息来解决这个问题。
4. 固定嵌入与学习嵌入：
SPE 是一种固定编码，这意味着它不是从数据中学习而来的，也不包含任何可训练的参数。
另一方面，学习嵌入可以更加灵活，并在训练过程中适应数据的特定特性。
然而，学习嵌入可能会对未知序列长度泛化能力较差，尤其是在使用相对较短的序列进行训练时。
本质上，虽然 SPE 是一种有价值的位置编码技术，但必须意识到它的局限性，尤其是在泛化到较长序列以及相对位置编码的隐式特性方面。为了解决这些缺陷，研究者探索了各种替代方案，例如学习嵌入和相对位置编码方法。

# 4 Why Self-Attention
![](https://upload-images.jianshu.io/upload_images/25141709-dbb665ad1633caa8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在n 和 d相当的情况下，不同类型的层计算复杂度是相差不大的，但是自注意力层通过恒定数量的顺序执行操作连接所有位置，使得计算复杂度只有O(1)而循环层则需要O(n)个顺序操作
# 5 Training
## 5.1  Training Data and Batching
数据集：
 - English-German: standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs, BPE,  37000 tokens
 - English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary
Batch:
句子对按近似序列长度分组。每个训练批次包含一系列句子对，其中包含大约 25,000 个源tokens和 25,000 个目标tokens。
##  5.2 Hardware and Schedule
8个 P100 GPU
transformer- base    每个step0.4s   训练了10w步 12hours
transformer-big    每个step 1s  训练了30w步 3.5days

## 5.3 optimizer
 Adam optimizer  with β1 = 0.9, β2 = 0.98 and ϵ = 10−9， Adam本身对学习率不敏感，学习率是根据公式自动计算出来的：
$$
 lrate = d^{−0.5}_{model} · min(step\_num^{−0.5},step\_num · warmup\_steps^{−1.5})
$$
在前 warmup_steps 个训练步骤中线性增加学习率，然后根据步数的平方根倒数按比例降低学习率。我们使用了 warmup_steps = 4000。

##  5.4 Regularization正则化
- residual dropout：在每个子层中，进入残差链接之前\进入layernorm之前还有 每个 embedding + PE 的过程都使用了dropout，dropout=0.1， 把10%的权重置成0，剩下百分之九十乘以1/(1-0.1)
-  LabelSmoothing: 训练期间 设置ϵls=0.1.在用softmax学习一个东西的时候，标号正确的是1，错误的是0，但是softmax是一个指数，很难逼近于1或者0，在这种情况下，我们认为ϵls达到一定值就足够了，就是置信度，比如，本文用的就是>=(1-0.1 = 0.9)是正确的1，<=0.1就是错误的0.这种不确信度增加了accuracy和BLEU score.

# 6 Results
## 6.1 MachineTranslation
![image.png](https://upload-images.jianshu.io/upload_images/25141709-3434514f0af4d814.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
还把模型在另一个任务上做了比较，体现了一定的泛化性，就不再做过多赘述了，不是本文的重点。
# 7 Conclusion
- 提出了一个序列转导模型，是第一个只用attention机制，而不用RNN和CNN实现的。
- 在机器翻译的任务上表现非常不错；
- transformer 可以不仅仅用于文本，可以尝试用于video，图片等资源上。

参考资料：
[1] https://blog.csdn.net/m0_37605642/article/details/132866365