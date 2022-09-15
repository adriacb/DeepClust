from utils import *
from ann import Autoencoder

from rdkit import Chem
from rdkit.Chem import Descriptors

def read_data():
    columns=['index', 'id', 'smile', 'logP', 'MWt', 'nHBD', 'NumAromaticRings', 'PFI',
    'fracCSP3', 'NumHAcceptors', 'NumHeteroatoms',
    'NumRotatableBonds', 'NumSaturatedRings', 
    'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NHOHCount', 'NOCount', 'NumRadicalElectrons',
    'NumValenceElectrons', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'NumAliphaticCarbocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles',
    'RingCount']
    df = pd.read_csv('/home/adria/TFM/data/zinc20_descriptors2.csv', names=columns)
    print(df.shape)
    print(df.head())


    selected = df[columns]
    selected = selected.loc[:,~selected.columns.duplicated()].copy()

    return selected


@timeit
def main():
    """Autoencoder and descriptors"""
    selected = read_data()
    #selected['RingCount'] = selected['smile'].apply(lambda x: Chem.Lipinski.RingCount(Chem.MolFromSmiles(x)))


    selected = selected.drop(['index', 'id', 'smile', 'MinAbsPartialCharge', 'MaxAbsPartialCharge'], axis=1)
    
    #check nan values
    print(selected.isna().sum())

    print(selected.info())

    # all to numeric
    selected = selected.apply(pd.to_numeric)
    
    with open('selected.pkl', 'wb+') as f:
        pickle.dump(selected, f)
    f.close()

    f2 = open('selected.pkl', 'rb+')
    selected = pickle.load(f2)#.iloc[:1000000,:]

    items_features_fitted = prepareInput2(selected)
    print(items_features_fitted)
    print(items_features_fitted.shape)

    length = items_features_fitted.shape.as_list()[1]    
    

    ac = Autoencoder(
                        input_dim=length, 
                        hidden_dim_enc=[20, 15],
                        hidden_dim_dec=[15, 20], 
                        output_dim=2
                    )

    autoencoder, encoder = ac.build_model()
    


    # split into training and testing sets (80/20 split)
    X_train, X_test = train_test_split(   
                            items_features_fitted.numpy(), 
                            test_size=0.2, random_state=42)

    
    

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)


    # fit the model using the training data
    history = autoencoder.fit(X_train, X_train, 
                                    epochs=50, batch_size=35, shuffle=True, 
                                    validation_data=(X_test, X_test),
                                    verbose=1,
                                    callbacks=[callback])
    

    with open('history.pkl', 'wb+') as f:
        pickle.dump(history, f)
    f.close()

    autoencoder.save('/home/adria/TFM/models/autoencoder3d.h5')
    encoder.save('/home/adria/TFM/models/encoder3d.h5')

    print(history)

    ac.plot_history(history.history, save=True, path='/home/adria/TFM/images/loss3d.png')
    ac.plot_accuracy(history.history, save=True, path='/home/adria/TFM/images/acc3d.png')
    ac.plot_model(path = '/home/adria/TFM/images/modelfinal_3d.png')

    
    

    # evaluate the model using the test data
    losss = autoencoder.evaluate(X_test, X_test)
    print('Test loss:', losss)

    predauto = encoder.predict(X_train)
    
    print(predauto)

def main2():
    """Autoencoder and descriptors"""
    selected = read_data()
    #selected['RingCount'] = selected['smile'].apply(lambda x: Chem.Lipinski.RingCount(Chem.MolFromSmiles(x)))


    selected = selected.drop(['index', 'id', 'smile', 'MinAbsPartialCharge', 'MaxAbsPartialCharge'], axis=1)
    
    #check nan values
    print(selected.isna().sum())

    print(selected.info())

    # all to numeric
    selected = selected.apply(pd.to_numeric)
    
    with open('selected.pkl', 'wb+') as f:
        pickle.dump(selected, f)
    f.close()

    f2 = open('selected.pkl', 'rb+')
    selected = pickle.load(f2)#.iloc[:500000,:]
    #selected = selected.drop_duplicates()

    print("--->",type(selected))


    items_features_fitted = prepareInput(selected)
    
    print(items_features_fitted)
    print(items_features_fitted.shape)

    length = items_features_fitted.shape.as_list()[1]    
    

    ac = Autoencoder(
                        input_dim=length, 
                        hidden_dim_enc=[20, 15],
                        hidden_dim_dec=[15, 20], 
                        output_dim=3
                    )

    autoencoder, encoder = ac.build_model2()
    


    # split into training and testing sets (80/20 split)
    X_train, X_test = train_test_split(   
                            items_features_fitted.numpy(), 
                            test_size=0.2, random_state=42)

    
    

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)


    # fit the model using the training data
    history = autoencoder.fit(X_train, X_train, 
                                    epochs=50, batch_size=64, shuffle=True, 
                                    validation_data=(X_test, X_test),
                                    verbose=1,
                                    callbacks=[callback])
    

    with open('history3d.pkl', 'wb+') as f:
        pickle.dump(history, f)
    f.close()

    autoencoder.save('/home/adria/TFM/models/autoencoder3d.h5')
    encoder.save('/home/adria/TFM/models/encoder3d.h5')

    print(history)

    ac.plot_history(history.history, save=True, path='/home/adria/TFM/images/loss_maxmin3d.png')
    ac.plot_accuracy(history.history, save=True, path='/home/adria/TFM/images/acc_maxmin3d.png')
    ac.plot_model(path = '/home/adria/TFM/images/modelfinal3d.png')

    
    

    # evaluate the model using the test data
    losss = autoencoder.evaluate(X_test, X_test)
    print('Test loss:', losss)

    predauto = encoder.predict(X_train)
    
    print(predauto)
    print(np.unique(predauto[:,1]))
    print(np.unique(predauto[:,0]))

if __name__ == '__main__':
	main2()
