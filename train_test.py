import torch
from torch import nn
from utilz.utils import *
from models.modules import *

def train_model(model, optimizer, criterion, scheduler, train_loader, test_loader, config):
    print("Training the model ...")
    patience_limit = config['patience_limit']
    sos = config['sos']
    eos = config['eos']
    out_dict_size = config['output_dict_size']
    future = config['future']// config['dwn_smple']
    train_loss, prev_average_loss,prev_test_ADE, patience= [],  10000000.0, 100000000.0, 0
    trainFDE, trainADE = [], []
    Best_Model = []
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses, epoch_ADE, epoch_FDE = [], [], []
        for Scene, Target, Adj_Mat_Scene in train_loader: # Scene & Taget => [B, SL0, Nnodes, Features], Adj_Mat_Scene => [B, SL, Nnodes, Nnodes]
            optimizer.zero_grad()
            # Let's add some noise to the Scene
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config)
            outputs = model(Scene, Scene_mask, Adj_Mat, Target[:,:-1]) # Remove the eos token for the input to the decoder
            loss = criterion(outputs.reshape(-1, out_dict_size), Target[:,1:].permute(0,2,1,3).reshape(-1).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            with torch.no_grad():
                _, ADE, FDE = Find_topk_selected_words(outputs[:,:-1].reshape(-1, config['Nnodes'], future, 2, out_dict_size).permute(0,2,1,3,4), Target[:,1:-1]) # sos and eos are not included in the loss
            epoch_losses.append(loss.item())
            epoch_ADE.append(ADE)
            epoch_FDE.append(FDE)

        trainADE.append(sum(epoch_ADE)/ len(train_loader))
        trainFDE.append(sum(epoch_FDE)/ len(train_loader))
        train_loss.append(sum(epoch_losses) / len(train_loader))
        scheduler.step()
        
        

        log = f'Epoch [{epoch+1}/{config['epochs']}], Loss: {train_loss[-1]:.2f}, ADE: {trainADE[-1]:.3f}, FDE: {trainFDE[-1]:.2f}'
        savelog(log, config['ct'])
        # Saving the best model to the file
        if train_loss[-1] < prev_average_loss: # checkpoint update
            prev_average_loss = train_loss[-1]  # Update previous average loss
            patience = 0
            # Best_Model = model
        elif patience > patience_limit:
                savelog(f'early stopping, Patience lvl1 , lvl2 {patience}', config['ct'])
                break
        patience += 1
        if config['Test_during_training'] and epoch % 2 == 1:
            ADE, FDE, _ = test_model(model, test_loader, config)
            savelog(f"During Training, Test ADE: {ADE :.2f}, FDE: {FDE :.2f}", config['ct'])
            model.train()
            if ADE < prev_test_ADE: # checkpoint update
                prev_test_ADE = ADE  # Update previous average loss
                patience = 0
                Best_Model = model
            # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.5f} MB")
            # print(f"Reserved memory:  {torch.cuda.memory_reserved() / 1024**2:.5f} MB")
    return Best_Model, train_loss, trainADE, trainFDE


def test_model(model, test_loader, config):
    sos = config['sos']
    eos = config['eos']
    future = config['future']// config['dwn_smple']
    savelog("Starting testing phase", config['ct'])
    model.eval()
    Avg_ADE, Avg_FDE, test_size = 0, 0, 0
    saved_buffer = torch.empty((0, future, config['Nnodes'], 2), device=config['device'])
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for Scene, Target, Adj_Mat_Scene in test_loader:
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config, test=True)
            Pred_target = greedy_search(model, Scene, Scene_mask, Adj_Mat, config, future)
            ADE, FDE = Find_topk_selected_words(Pred_target[:,1:-1].reshape(-1, config['Nnodes'], future, 2).permute(0,2,1,3), Target[:,1:-1], True) # sos and eos are not included in the loss
            test_size += Scene.size(0)
            Avg_ADE += ADE
            Avg_FDE += FDE
            if config['Save_Predictions']:
                saved_buffer = torch.cat((saved_buffer, Pred_target[:,1:-1].reshape(-1, config['Nnodes'], future, 2).permute(0,2,1,3)), dim=0)
        end_event.record()
        torch.cuda.synchronize()
        Avg_inference_time = start_event.elapsed_time(end_event)/test_size
        Avg_ADE, Avg_FDE = Avg_ADE/len(test_loader), Avg_FDE/len(test_loader)
        log= f"ADE is : {Avg_ADE:.3f} px \n FDE is: {Avg_FDE:.3f} px \n Inference time: {1000*Avg_inference_time:.3f} ms"
        savelog(log, config['ct'])
        return Avg_ADE, Avg_FDE, saved_buffer
    
def prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config, test=False):
    if not test:
        noise_mask = torch.randn_like(Scene[...,:2]) < config['noise_ratio']
        mask = Scene[...,:2] != 0  # Mask to avoid noise on the zero values
        noise_mask = noise_mask & mask
        Scene[...,:2] = Scene[...,:2] + noise_mask * torch.randint(0, config['noise_amp'], Scene[...,:2].shape, device=Scene.device)
        # Same for the Target
        noise_mask = torch.randn_like(Target[...,:2]) < config['noise_ratio']
        mask = Target[...,:2] != 0  # Mask to avoid noise on the zero values
        noise_mask = noise_mask & mask
        Target[...,:2] = Target[...,:2] + noise_mask * torch.randint(0, config['noise_amp'], Target[...,:2].shape, device=Target.device)
    Scene = attach_sos_eos(Scene, sos, eos,Scene.shape[1],Target.shape[1]) # Scene => [B, SL0+2, Nnodes, Features]
    Adj_Mat = torch.cat((torch.ones_like(Adj_Mat_Scene[:,:1]), Adj_Mat_Scene, torch.ones_like(Adj_Mat_Scene[:,:1])), dim=1)
    Scene_mask = create_src_mask(Scene)
    Target = attach_sos_eos(Target[:,:,:, config['xy_indx']],sos[:,config['xy_indx']], eos[:,config['xy_indx']], Scene.shape[1],Target.shape[1])
    return Scene, Scene_mask, Adj_Mat, Target

def greedy_search(model, scene, scene_mask, adj_mat, config, max_len):
    model.eval()
    B, N = scene.size(0), scene.size(2)
    pred_trgt = config['sos'][0,config['xy_indx']].repeat(B*N,1,1)
    enc_out = model.encoder(scene, scene_mask, adj_mat)
    for _ in range(max_len+1):  # +1 to include the last prediction
        trg_mask = target_mask(pred_trgt, num_head=model.num_heads, device=scene.device)
        dec_out = model.decoder(pred_trgt, enc_out, trg_mask, scene_mask)
        proj = model.proj(dec_out)
        top1 = proj.softmax(-1).argmax(-1) # Get the index of the max value
        pred_trgt = torch.cat((pred_trgt, top1[:,-1:]), dim=1)  # Append the last prediction to the target sequence
    return pred_trgt
if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")