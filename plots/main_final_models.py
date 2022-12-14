import sys
import submodels_module as modelbank


def main():
    '''
    current model options are [ridge,forest,svm,fnn,emb_fnn_flat,emb_fnn_maxpool,emb_rnn,emb_cnn]
    '''


    toggle_no=int(sys.argv[1])
    if toggle_no<100:
        sample_size=1
    elif toggle_no<200:
        sample_size=0.5
        toggle_no=toggle_no-100
    elif toggle_no<300:
        sample_size=0.25
        toggle_no=toggle_no-200

    
    if toggle_no<12:
        ### seq_to_yield model using measured yields
        a_models=['ridge','forest','svm','fnn','emb_fnn_maxpool','emb_fnn_flat','emb_rnn','emb_cnn','small_emb_rnn','small_emb_atn_rnn','small_emb_cnn','small_emb_atn_cnn']
        # a=modelbank.seq_to_yield_model(a_models[toggle_no],1)
        a=modelbank.final_seq_to_yield_model(a_models[toggle_no],sample_size)
        a.cross_validate_model()
        a.test_model()
        a.plot()


    # ### assay_to_yield_model
    # b=modelbank.assay_to_yield_model([1,8,9,10],'forest',1)
    # b.cross_validate_model()
    # b.test_model()
    # b.plot()
    # b.save_predictions() 

    ### use predictions of yield from model a to make a seq-to-(predicted)yield model
    # c_models=['ridge','fnn','emb_fnn_maxpool','emb_fnn_flat','emb_rnn','emb_cnn']
    # for j in range(1):
    #     assay_to_yield_model_no=3 #for each saved model from a
    #     c=modelbank.seq_to_pred_yield_model([[1,8,9,10],'forest',1,assay_to_yield_model_no],[c_models[toggle_no],1])
    #     c.cross_validate_model()
    #     c.test_model()
    #     c.plot()




    ### create a sequence to assay model
    # d_models=['ridge','fnn','emb_fnn_maxpool','emb_fnn_flat','emb_rnn','emb_cnn','small_emb_rnn','small_emb_atn_rnn','small_emb_cnn','small_emb_atn_cnn']
    # d=modelbank.seq_to_assay_model([1,8,9,10],d_models[toggle_no],1)
    # for i in range(10):
    #    d.cross_validate_model()
    # d.test_model()  
    # d.plot()
    # d.save_predictions() 
    # if toggle_no >1:
    #     d.save_sequence_embeddings() 

    ###use assay predictions of test set and assay_to_yield model 
    # for j in range(3):
    #     seq_to_assay_model_no=j
    #     b.apply_predicted_assay_scores([d_models[toggle_no],1,seq_to_assay_model_no])


    if toggle_no>=12:
        toggle_no=toggle_no-12
        ##use sequence embeddings to predict yield
        embeddings_list=['emb_fnn_flat','emb_fnn_maxpool','emb_rnn','emb_cnn','small_emb_rnn','small_emb_atn_rnn','small_emb_cnn','small_emb_atn_cnn']
        e_models=['ridge','forest','svm','fnn']
        emb_index=int(toggle_no/4)
        model_index=toggle_no-(emb_index*4)
        for j in range(3):
            seq_to_assay_model_no=j
        #     e=modelbank.sequence_embeding_to_yield_model([[1,8,9,10],embeddings_list[emb_index],1,seq_to_assay_model_no],e_models[model_index],1)
            e=modelbank.final_sequence_embeding_to_yield_model([[1,8,9,10],embeddings_list[emb_index],1,seq_to_assay_model_no],e_models[model_index],sample_size)
            e.cross_validate_model()
            e.test_model()
            e.plot()

if __name__ == '__main__':
    main()