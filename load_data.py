from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json
import clip

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

class PACSDatasetDisentangle(Dataset):
    """A Dataset that is composed both of source and target samples.
    It returns two elements, one from source and one from target"""
    
    def __init__(self, source_samples, target_samples, transform):
        self.source_samples = source_samples
        self.target_samples = target_samples
        self.src_len = len(source_samples)
        self.trg_len = len(target_samples)
        self.len = max(self.src_len, self.trg_len)
        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        img_path_src, y_src = self.source_samples[index % self.src_len]
        img_path_trg, y_trg = self.target_samples[index % self.trg_len]
        
        x_src = self.transform(Image.open(img_path_src).convert('RGB'))
        x_trg = self.transform(Image.open(img_path_trg).convert('RGB'))

        return x_src, y_src, x_trg, y_trg

class PACSDatasetCLIP1(Dataset):
    """A Dataset that is composed both of source and target samples.
    It returns two elements, one from source and one from target"""
    
    def __init__(self, source_samples, target_samples, transform):
        self.source_samples = source_samples
        self.target_samples = target_samples
        self.src_len = len(source_samples)
        self.trg_len = len(target_samples)
        self.len = max(self.src_len, self.trg_len)
        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        src, y_src = self.source_samples[index % self.src_len]
        img_path_src, d_src = src['image_name'], src['descriptions']
        trg, y_trg = self.target_samples[index % self.trg_len]
        img_path_trg, d_trg = trg['image_name'], trg['descriptions']
        
        x_src = self.transform(Image.open(img_path_src).convert('RGB'))
        x_trg = self.transform(Image.open(img_path_trg).convert('RGB'))

        return x_src, y_src, d_src, x_trg, y_trg, d_trg

class PACSDatasetCLIP2(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, d = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, d

def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples

def read_lines_clip(data_path, domain_name):
    labeled_examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()
    
    with open(f'labeled_PACS/labeled_{domain_name}.txt') as g:
        labeled_lines = g.readlines()
        
    labeled_image_names = []
    descriptions = []
        
    for labeled_line in labeled_lines:
        d = json.loads(labeled_line.strip())
        labeled_image_names.append(d['image_name'])
        descriptions.append(d['descriptions'])

    for line in lines:
        descr = ''
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        
        if f'{domain_name}/{category_name}/{image_name}' in labeled_image_names:
            descr = descriptions[labeled_image_names.index(f'{domain_name}/{category_name}/{image_name}')]
            
        img = {'image_name': image_path, 'descriptions': descr} 
        if category_idx not in labeled_examples.keys():
            labeled_examples[category_idx] = [img]
        else:
            labeled_examples[category_idx].append(img)
                
    return labeled_examples

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_domain_disentangle(opt):
    """Return DataLoaders for the domain disentangle experiment."""
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    # how to split the samples??
    # since the target is used at traing time without the label
    # it makes sense to use it as the test set...
    # I will ask the professor
    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)
    
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length_source = source_total_examples * 0.2 # 20% of the training split used for validation
    val_split_length_target = target_total_examples * 0.2 # 20% of the training split used for validation

    source_samples = []
    target_samples = []
    val_samples = []
    test_samples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length_source)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                source_samples.append([example, category_idx]) 
            else:
                val_samples.append([example, category_idx]) 

    for category_idx, examples_list in target_examples.items():
        split_idx = round(target_category_ratios[category_idx] * val_split_length_target)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                target_samples.append([example, category_idx]) 
            else:
                val_samples.append([example, category_idx]) 
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_samples.append([example, category_idx]) 

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    # Dataloaders
    train_loader = DataLoader(PACSDatasetDisentangle(source_samples, target_samples, train_transform), batch_size=32, num_workers=1, shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_samples, eval_transform), batch_size=32, num_workers=1, shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_samples, eval_transform), batch_size=32, num_workers=1, shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_clip_disentangle(opt):
    """Return DataLoaders for the CLIP domain disentangle experiment."""
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    # how to split the samples??
    # since the target is used at traing time without the label
    # it makes sense to use it as the test set...
    # I will ask the professor
    source_examples = read_lines_clip(opt['data_path'], source_domain)
    target_examples = read_lines_clip(opt['data_path'], target_domain)
    
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length_source = source_total_examples * 0.2 # 20% of the training split used for validation
    val_split_length_target = target_total_examples * 0.2 # 20% of the training split used for validation

    source_samples = []
    target_samples = []
    val_samples = []
    test_samples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length_source)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                source_samples.append([{'image_name': example['image_name'],'descriptions': str(example['descriptions'])}, category_idx ]) 
            else:
                val_samples.append([example['image_name'], category_idx]) 

    for category_idx, examples_list in target_examples.items():
        split_idx = round(target_category_ratios[category_idx] * val_split_length_target)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                target_samples.append([{'image_name': example['image_name'],'descriptions': str(example['descriptions'])}, category_idx ])
            else:
                val_samples.append([example['image_name'], category_idx]) 
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_samples.append([example['image_name'], category_idx]) 

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    # Dataloaders
    train_loader = DataLoader(PACSDatasetCLIP1(source_samples, target_samples, train_transform), batch_size=32, num_workers=1, shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_samples, eval_transform), batch_size=32, num_workers=1, shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_samples, eval_transform), batch_size=32, num_workers=1, shuffle=False)

    return train_loader, val_loader, test_loader
