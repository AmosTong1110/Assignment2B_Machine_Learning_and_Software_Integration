"""Train a model on a prepared 15-minute traffic series."""
import argparse
from train import train_and_evaluate
from data_processing import load_site_series
from models import predict_next

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Prepared traffic CSV')
    parser.add_argument('--site', required=True, help='SCATS site ID')
    parser.add_argument('--models', default='models')
    parser.add_argument('--model', choices=['linear','lstm','gru'], default='linear')
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    args=parser.parse_args()
    result=train_and_evaluate(args.data,args.site,args.model,args.models,window=args.window,epochs=args.epochs)
    _,history=load_site_series(args.data,args.site)
    print('Site:',args.site,'model:',args.model)
    print('Test period:',result['test_start'],'to',result['test_end'])
    print('MAE:',round(result['mae'],3),'RMSE:',round(result['rmse'],3),result['unit'])
    print('Next interval prediction:',round(predict_next(args.models,args.model,args.site,history),3),result['unit'])
    print('Third model requires tutor approval. Do not tune repeatedly on the final test data.')
if __name__=='__main__':
    main()
