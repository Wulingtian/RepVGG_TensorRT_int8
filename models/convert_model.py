def main():
    #gen coco pretrained weight
    import torch
    num_classes = 2
    checkpoint = torch.load("RepVGG-A0-train.pth")
    #print(checkpoint.keys())
    for key in list(checkpoint.keys()):
        checkpoint["module." + key] = checkpoint.pop(key)

    checkpoint["module.linear.weight"] = checkpoint["module.linear.weight"][:num_classes, :]
    checkpoint["module.linear.bias"] = checkpoint["module.linear.bias"][:num_classes]
    torch.save(checkpoint, "RepVGG-A0-classes%d.pth" % num_classes)

    # weight
    #model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][:num_classes, :]
    #model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][:num_classes, :]
    #model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][:num_classes, :]
    # bias
    #model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
    #model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
    #model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]
    # save new model
    #torch.save(model_coco, "cascade_rcnn_r50_coco_pretrained_weights_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    main()

