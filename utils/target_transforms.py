import torch
import torchvision.transforms as transforms

class TargetResize(object):
    def __init__(self, input_size, output_size):
        assert isinstance(input_size, (int, tuple))
        assert isinstance(output_size, (int, tuple))
        self.input_size = input_size
        self.output_size = output_size
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.output_size),
            transforms.ToTensor()
        ])
        
    def __call__(self, target):
        if isinstance(self.input_size, int):
            h, w = self.input_size, self.input_size
        else:
            h, w = self.input_size[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_target = {
            'boxes': [],
            'labels': target['labels'].clone()}
        if 'masks' in target.keys():
            new_target['masks'] = self.mask_transform(target['masks'].clone())
        if 'scores' in target.keys():
            new_target['scores'] = target['scores'].clone()
                
        for i, box in enumerate(target['boxes']):
            w_ratio = new_w / w
            h_ratio = new_h / h
            new_target['boxes'].append(
                box * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio]))
        if new_target['boxes']:
            new_target['boxes'] = torch.stack(new_target['boxes'])
        
        return new_target