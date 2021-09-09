import numpy as np
import torch
import torchvision.transforms as transform

def self_ensemble(img, net):
    outputs = []
    for k in range(0, 8, 1):
        # Rotate 90*k degrees and mirror flip when k>=4
        test_input = np.rot90(img, k) if k < 4 else np.fliplr(np.rot90(img, k))

        test_input = (torch.Tensor(test_input.copy()).permute(2, 0, 1)).unsqueeze_(0)
        test_input = test_input.cuda()

        # Apply network on the rotated input
        tmp_output, lr_cyc,_ = net(test_input)
        tmp_output = np.clip(np.squeeze(tmp_output.cpu().detach().permute(0, 2, 3, 1).numpy()), 0, 1)

        # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
        tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)
        # fix SR output with back projection technique for each augmentation

        # save outputs from all augmentations
        outputs.append(tmp_output)

    # Take the median over all 8 outputs
    final_sr = np.median(outputs, 0)

    final_sr = (torch.Tensor(final_sr.copy()).permute(2, 0, 1)).unsqueeze_(0)
    # print('final_sr', final_sr.shape)
    return final_sr, lr_cyc
