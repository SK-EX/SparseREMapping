### project dir 
        ---./depth
                depth.py {用于处理输入的图像深度值}
        ---./utils
                dataloader.py(数据加载)
                mlp.py(如有需要,网络中扩展mlp)
                attention.py(用于提高图像注意力模块,特别是桩周的地形深度特征)
                transform.py(如有需要,网络中扩展transform)
                camerapose.py(如果需要,计算每个图像的相机位姿,引入物理透视规则(参照vggsfm,colmap等算法架构),精确每张图像的相机位姿,需要从不同视角规范冲刷坑的形态,因为冲刷坑在不同视角是一致的,但是透视的形态会有所不同)
                loss.py(损失函数)
        ---./model
                model.py(网络架构,用于输出桩周的深度高程映射,沙面为高程正,冲刷坑为负,以掩模的形式渲染在原图中)
        ---./train
                train.py(主训练入口)
        ---./result
            ./result/checkpoints(训练结果)
            ./result/loss
        ---./data
        ---./config
        ---./visulize(训练结果可视化,多视角下冲刷坑形态深度-高程)
模型训练的内容是输入单桩-冲刷床面的图像,网络需要训练RGB图像中的单桩周围的冲刷坑形态.以及冲刷坑相对于周围沙面的深度高程映射(输出在原图上,用颜色条表示,其次是输出冲刷坑形态)