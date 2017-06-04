# GAN 实现及相关设计

## Paddle支持 sub-graph

GAN 实现时，需要多个子图分开训练，tf相关模型如下：

```python
with tf.Session() as session:
    tf.initialize_all_variables().run()

    for step in xrange(num_steps):
        # update discriminator
        x = data.sample(batch_size)
        z = gen.sample(batch_size)
        session.run([loss_d, opt_d], {
            x: np.reshape(x, (batch_size, 1)),
            z: np.reshape(z, (batch_size, 1))
        })

        # update generator
        z = gen.sample(batch_size)
        session.run([loss_g, opt_g], {
            z: np.reshape(z, (batch_size, 1))
        })
```

但目前 Paddle 似乎还不支持子图单独训练，现有 GradientMachine 耦合了parameter, layer资源管理和计算，而子图训练需要将计算和参数, layer分开。

参考了下之前 [gan](https://github.com/PaddlePaddle/Paddle/blob/068bfbb817611c856acd8c535de2b33a6126786c/demo/gan/gan_trainer.py)，
打算使用类似的操作 `swig_paddle` 的方式，但为了优雅一些，在v2 中用python支持类似 `tf.Session` 的方法。

需要实现的功能如下：

```python
from paddle.v2.trainer import Session # maybe another name is better

# create a global GradientMachine to create layers and init all parameters
with Session() as sess:

    for pass_id in range(pass):
        for batch_id, batch in enumerate(data_provider()):

            # each sess.run() will create a small GradientMachine before the first time exectuation(may
            # use a global cache to avoid to be duplicately created),
            # this small GradientMachine will create a small Topoloty for the sub-graph, and copy
            # all shared parameters from the glboal GradientMachine to ensure all the sub-graphs
            # share the global model's parameters.
            cost1, model1_output = sess.run([model1_cost, model1_output], {'data1': batch[0]})

            # the `reader` is discarded, and data is feeded batch by batch so that we can provide
            # one or more sub-graphs' outputs as another sub-graph's inputs.
            cost2 = sess.run([model2_cost], {'model1_output': model1_output})
```

1. 添加新 v2 接口，从python 层拆分GradientMachine的资源和计算
  1. 提供类似 `tf.Session` 创建全局的GradientMachine，用来管理所有的 layer, paramerter
  2. 提供类似 `sess.run` 的方法，实现类似  `sess.run([trgs])` 包含的子图计算
     - 每个 sess.run 会根据 inputs 构建一个全局唯一的 GradientMachine 来进行子图计算
     - 子图的 GradientMachine 会share全局 GradientMachine的 parameter
  3. 数据按 batch 输入，来增加 input 的多样性，比如一个子图的input可以是其他子图的output
