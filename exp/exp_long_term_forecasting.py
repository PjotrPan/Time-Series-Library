from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _weighted_loss(self, pred, true, cpu=False):
        weight = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]*true.shape[0]).reshape(true.shape)
        weight = weight.float()
        if not cpu:
            weight = weight.to(self.device)
        loss = (weight * (pred - true)**2).mean()
        return loss

    def setup_sweep(self, args):
        print("\n start sweeping \n")

        sweep_configuration = {
            'method': 'grid',
            'name': 'sweep',
            'metric': {'goal': 'minimize', 'name': 'Test MAE'},
            'parameters': 
            {
                'd_ff': {'values': [32, 64]},
                'd_model': {'values': [32, 64]},
                #'e_layers': {'values': [6, 8, 10, 16]},
                #'d_layers': {'values': [2, 8]},
                #'n_heads': {'values': [2, 3, 4]},
                #'model': {'values': ["Pyraformer", "Transformer", "Informer", "Reformer", "Autoformer"]},
                #'patient_numbers': {'values': [[540],[544],[552],[567],[584],[596]]}
            }
        }

        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project="glucose_prediction"
        )

        wandb.agent(sweep_id, function = self.train )

    def update_sweep(self):       
        #self.args.seq_len = wandb.config.seq_len
        #self.args.scaler = wandb.config.scaler
        #self.args.interpolation = wandb.config.interpolation
        #self.args.filter_size = wandb.config.filter_size
        self.args.d_ff = wandb.config.d_ff
        self.args.d_model = wandb.config.d_model
        #self.args.e_layers = wandb.config.e_layers
        #self.args.d_layers = wandb.config.d_layers
        #self.args.n_heads = wandb.config.n_heads
        #self.args.model = wandb.config.model
        #self.args.patient_numbers = wandb.config.patient_numbers

        self.args.enc_in = 1 + len(self.args.features) #dont change

        self.model = self._build_model().to(self.device)

        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.task_name,
            self.args.model_id,
            self.args.model,
            self.args.data,
            self.args.features,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, 1)
        return setting

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting=None):
        run = wandb.init(project="glucose_prediction")
        if self.args.sweep:
            setting = self.update_sweep()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    #loss = self._weighted_loss(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = 1

            wandb.log({"training loss": train_loss, "validation loss": vali_loss})

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #if self.args.sweep == True:
        test_mae, rmse = self.test(setting)
        wandb.log({'Test MAE': test_mae})
        wandb.log({'Test RMSE': rmse})

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        from tqdm import tqdm
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):

                if True in torch.isnan(batch_x) or True in torch.isnan(batch_y):
                    continue
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred[:,-1,:])
                trues.append(true[:,-1,:])

        preds = np.array(preds)
        trues = np.array(trues)

        # result save
        model_description = f"{self.args.model}_{int(self.args.patient_numbers[0])}_{self.args.data_path}_predlen{self.args.pred_len}_forecasthorizon{self.args.seq_len}_scaler{self.args.scaler}_interpolation{self.args.interpolation_method}_d_model{self.args.d_model}_d_ff{self.args.d_ff}_e_layers{self.args.e_layers}_d_layers{self.args.d_layers}_n_heads{self.args.n_heads}"
        folder_path = './results/' + model_description + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = test_data.inverse_transform(preds.reshape(-1,1))
        trues = test_data.inverse_transform(trues.reshape(-1,1))

        p = preds.copy()
        t = trues.copy()
        np.save(folder_path + "preds", p)
        np.save(folder_path + "trues", t)
        t1 = wandb.Table(data=p.reshape(-1,1), columns=['value'])
        t2 = wandb.Table(data=t.reshape(-1,1), columns=['value'])
        wandb.log({'preds': t1})
        wandb.log({'trues': t2})      

        mae = np.abs(t - p).mean()
        mse = ((t-p)**2).mean()
        rmse = np.sqrt(mse)
        np.save(folder_path + "mae", mae)
        np.save(folder_path + "rmse", rmse)
        print(f'{"="*60}\n\nMAE: {mae}\nRMSE: {rmse} \n{"="*60}')
        
        #f = open(folder_path + "result_long_term_forecast.txt", 'a')
        #f.write(setting + "  \n")
        #f.write('mae:{}, rmse:{}'.format(mae, rmse))
        #f.write('\n')
        #f.write('\n')
        #f.close()

        return mae, rmse
