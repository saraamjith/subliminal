"""Code of "Subliminal Learning Crosses Modalities". The module of each step, by section of the paper:

    Sec. 3    prep_images, teacher_data, train_student, base_states
    Sec. 4    image_readout, answers, image_shift, text_preference
    Sec. 5    gen_embed, gen_images, gen_score, gen_analysis
    Sec. 6.1  constant_vector, constant_vector_analysis
    Sec. 6.2  weight_lens
    shared    config (paths, models, traits), common (models, evaluation sets, hooks), geometry (directions, image shift, lens)

`python -m sl <command>` runs one step (sl/__main__.py).
"""
