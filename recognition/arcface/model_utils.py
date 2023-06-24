import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101

class arcface_classifier(torch.nn.Module):
    def __init__(self, n_classes, margin, radius, model_type, embedding_dimension=2):
        super(arcface_classifier, self).__init__()

        # Create attributes
        self.classes = n_classes
        self.margin = margin
        self.radius = radius
        self.embed_dimension = embedding_dimension
        self.epsilon = 1e-8
        
        # Get the backbone and corresponding embed dimension of the same
        self.backbone, in_feats = self.get_backbone(model_type)

        # Modify the fully connected layer of the backbone to project in a smaller dimension
        self.backbone.fc = nn.Linear(
            in_features=in_feats, out_features=embedding_dimension
        )
        self.bn = nn.BatchNorm1d(embedding_dimension)

        # Classification layer
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embedding_dimension))
        nn.init.xavier_uniform_(self.weight)
    
    def get_backbone(self, model_type):
        # Projection Layers
        if model_type == "resnet18":
            backbone = resnet18(); in_feats = 512
        elif model_type == "resnet34":
            backbone = resnet34(); in_feats = 512
        elif model_type == "resnet50":
            backbone = resnet50(); in_feats = 2048
        elif model_type == "resnet101":
            backbone = resnet101(); in_feats = 2048
        return backbone, in_feats

    def get_embedding(self, x):
        # Pass through the backbone and normalize the output
        res_output = self.backbone(x)
        normed_output = self.bn(res_output)
        return res_output, normed_output

    def adjust_angles(self, cosine, labels):
        
        old_cosine = cosine
        
        # Ensure that the cosines are in the valid range of -1 to 1 and take an arccos to obtain the angle
        arc_cos = torch.clamp(cosine, -1 + self.epsilon, 1 - self.epsilon).arccos()

        # Extract the target angles separately
        batch_size = len(labels)
        target_angles = arc_cos[list(range(batch_size)), labels]
        new_target_angles = torch.clamp(target_angles + self.margin, self.epsilon, torch.pi - self.epsilon)

        # Add the margin to the target angles
        arc_cos[list(range(batch_size)), labels] = new_target_angles
        cosine = arc_cos.cos()
        
        if torch.any(cosine):
            # print(old_cosine)
            print(new_target_angles, labels)
            print(cosine)
            
        return cosine

    def forward(self, x, labels=None):
        # Extract the batch normed embedding
        _, embed = self.get_embedding(x)
        
        # Get the logit value
        logits = F.linear(F.normalize(embed), F.normalize(self.weight))
        print(f"LOGITS MAX: {logits.max()}; LOGITS MIN: {logits.min()}")

        # If labels are provided then add the margin to angle between respective
        # target center and embedding
        if labels is not None: logits = self.adjust_angles(logits, labels)

        # Project the result on a sphere of set radius and return the logits
        return self.radius * logits

# Test if the above model definition code works 
if __name__ == "__main__":
    t = torch.randn(10, 3, 112, 112)
    
    # Create a resnet50 backbone with 32 dimensional embedding space for face-vectors
    m = arcface_classifier(
        n_classes=24, margin=0.5, radius=10, model_type = "resnet50", embedding_dimension=32
    )

    print(m)
    print(m(t).shape)
    
    # Create a resnet18 backbone with 2 dimensional embedding space for face-vectors
    m = arcface_classifier(
        n_classes=24, margin=0.5, radius=10, model_type = "resnet18", embedding_dimension=2
    )
    
    import random
    labels = torch.tensor([random.randint(1, 5) for i in range(10)])

    print(m(t, labels).shape)
    print(m(t).shape)