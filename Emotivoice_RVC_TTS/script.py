# Credit to Diffusion_TSS for script.py, modifications are made to adapt to EMotivoice

import gc
import os
import sys
import traceback

import torch
import torchaudio

from pathlib import Path
import time
import wavio


# Credit to Diffusion_TSS for tts_preprocessor.py
sys.path.append(os.path.dirname(__file__))
import tts_preprocessor


# Emotive begin------------------------------------------------------------------
__EMOTI_TTS_DEBUG__ = False
import os, glob
import numpy as np
from yacs import config as CONFIG
import torch
import re

EMOTIVOICE_ROOT = os.path.dirname(__file__)
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'EmotiVoice')
RVC_DIR = os.path.join(os.path.dirname(__file__), 'rvc_gui')

if __EMOTI_TTS_DEBUG__:
    print(f'Emotivoice_RVC_TTS ROOT_DIR: {ROOT_DIR}')
sys.path.append(ROOT_DIR)

from .EmotiVoice.frontend import g2p_cn_en, g2p_id, read_lexicon, G2p
from .EmotiVoice.models.prompt_tts_modified.jets import JETSGenerator
from .EmotiVoice.models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer

from .rvc_gui import rvcgui_tts as rvc

import base64
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WAV_VALUE = 32768.0

# Copy over EmotiVoice.config.joint.config with new ROOT_DIR as we cannot use it due to path issues
def get_labels_length(file_path):
    """
    Return labels and their count in a file.

    Args:
        file_path (str): The path to the file containing the labels.

    Returns:
        list: labels; int: The number of labels in the file.
    """
    with open(file_path, encoding = "UTF-8") as f:
        tokens = [t.strip() for t in f.readlines()]
    return tokens, len(tokens)


class Config:
    #### PATH ####
    DATA_DIR            = ROOT_DIR + "/data/youdao/"
    train_data_path     = DATA_DIR + "train_am/datalist.jsonl"
    valid_data_path     = DATA_DIR + "valid_am/datalist.jsonl"
    output_directory    = ROOT_DIR + "/outputs"
    speaker2id_path     = DATA_DIR + "text/speaker2"
    emotion2id_path     = DATA_DIR + "text/emotion"
    pitch2id_path       = DATA_DIR + "text/pitch"
    energy2id_path      = DATA_DIR + "text/energy"
    speed2id_path       = DATA_DIR + "text/speed"
    bert_path           = ROOT_DIR + "/WangZeJun/simbert-base-chinese"
    token_list_path     = DATA_DIR + "text/tokenlist"
    style_encoder_ckpt  = ROOT_DIR + "/outputs/style_encoder/ckpt/checkpoint_163431"
    tmp_dir             = ROOT_DIR + "/tmp"
    model_config_path   = ROOT_DIR + "/config/joint/config.yaml"

    #### Model ####
    bert_hidden_size = 768
    style_dim = 128
    downsample_ratio    = 1     # Whole Model

    #### Text ####
    tokens, n_symbols = get_labels_length(token_list_path)
    sep                 = " "

    #### Speaker ####
    speakers, speaker_n_labels = get_labels_length(speaker2id_path)

    #### Emotion ####
    emotions, emotion_n_labels = get_labels_length(emotion2id_path)

    #### Speed ####
    speeds, speed_n_labels = get_labels_length(speed2id_path)

    #### Pitch ####
    pitchs, pitch_n_labels = get_labels_length(pitch2id_path)

    #### Energy ####
    energys, energy_n_labels = get_labels_length(energy2id_path)

    #### Train ####
    # epochs              = 10
    lr                  = 1e-3
    lr_warmup_steps     = 4000
    kl_warmup_steps     = 60_000
    grad_clip_thresh    = 1.0
    batch_size          = 16
    train_steps         = 10_000_000
    opt_level           = "O1"
    seed                = 1234
    iters_per_validation= 1000
    iters_per_checkpoint= 10000


    #### Audio ####
    sampling_rate       = 16_000
    max_db              = 1
    min_db              = 0
    trim                = True

    #### Stft ####
    filter_length       = 1024
    hop_length          = 256
    win_length          = 1024
    window              = "hann"

    #### Mel ####
    n_mel_channels      = 80
    mel_fmin            = 0
    mel_fmax            = 8000

    #### Pitch ####
    pitch_min           = 80
    pitch_max           = 400
    pitch_stats         = [225.089, 53.78]

    #### Energy ####
    energy_stats        = [30.610, 21.78]


    #### Infernce ####
    gta                 = False

config = Config()

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def get_models():
    
    am_checkpoint_path = scan_checkpoint(f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')

    style_encoder_checkpoint_path = scan_checkpoint(f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)#f'{config.output_directory}/style_encoder/ckpt/checkpoint_163431' 

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    model_CKPT = torch.load(style_encoder_checkpoint_path, map_location="cpu")
    model_ckpt = {}
    for key, value in model_CKPT['model'].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt, strict=False)
    generator = JETSGenerator(conf).to(DEVICE)

    model_CKPT = torch.load(am_checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(model_CKPT['generator'])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}


    return (style_encoder, generator, tokenizer, token2id, speaker2id)

def get_style_embedding(prompt, tokenizer, style_encoder):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]
    with torch.no_grad():
        output = style_encoder(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding

def tts(prompt, content, speaker, models):
    (style_encoder, generator, tokenizer, token2id, speaker2id)=models
    
    text = ""
    if params['rvc_language'] == "english_or_chinese":
        text =  g2p_cn_en(content, g2p, lexicon)
    else:
        text =  g2p_id(content)
    

    print(f"text emotivoice {text}")

    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding(content, tokenizer, style_encoder)

    speaker = speaker2id[speaker]

    text_int = [token2id[ph] for ph in text.split()]
    
    sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = torch.from_numpy(content_embedding).to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

    with torch.no_grad():

        infer_output = generator(
                inputs_ling=sequence,
                inputs_style_embedding=style_embedding,
                input_lengths=sequence_len,
                inputs_content_embedding=content_embedding,
                inputs_speaker=speaker,
                alpha=1.0
            )

    audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
    audio = audio.to(torch.int16).cpu()

    return audio

speakers = config.speakers
models = None
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()
# Emotive end------------------------------------------------------------------

def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still within the max length
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    return rv

current_params = None
params = {
    'activate': True,
    'output_dir': Path(EMOTIVOICE_ROOT + "/outputs"),
    'voice': speakers[0],
    'model_swap': False,
    'sentence_length': 20,
    'show_text': True,
    'autoplay': True,
    'disable_text_stream': False,
    'prompt': 'happy',
    'rvc_language': 'english_or_chinese',
    'rvc_model': '',
    'rvc_model_index': '',
    'rvc_method_entry': 'crepe',
    'rvc_pitch': 0,
    'rvc_crepe_hop': 128,
    'rvc_retrieval_rate': 0.4
}

controls = {}


def load_model():
    global models
    # Init TTS
    try:
        if models:
            for model in models:
                model = model.to(DEVICE)
        else:
            models = get_models()
    except Exception as e:
        print(e)
        models = None


def unload_model():
    global models
    try:
        if models:
            for model in models:
                model = model.to('cpu')
    except Exception as e:
        print(e)
        models = None


def remove_tts_from_history(history):
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]

    return history

def clear_output_dir():
    for f in os.listdir(params['output_dir']):
        if f.endswith(".wav"):
            if __EMOTI_TTS_DEBUG__:
                print(f"Emotivoice_RVC_TTS Removing {f}")
            os.remove(os.path.join(params['output_dir'], f))

def toggle_text_in_history(history):
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]

    return history

# Modifying chat state
# TTS usually turns assistant output streamming off, in order to prevent the user from seeing the assistant's output w/o corresponding tts audio
def state_modifier(state):
    if not params['activate']:
        return state
    
    if __EMOTI_TTS_DEBUG__:
        print(f"Emotivoice state_modifier: \n{state}")

    if params['disable_text_stream']:
        state['stream'] = False
        
    return state

# Modifying user inputs messages, 
# Atcually there is nothing to modify, but just spitting out "during processing message" in the assistant message box
def input_modifier(string, state):
    if not params['activate']:
        return string

    if __EMOTI_TTS_DEBUG__:
        print(f"Emotivoice input_modifier: \n{string}")

    print("*Emotivoice is generating text to speech...*")
    return string


def history_modifier(history):
    if not params['activate']:
        return history
    
    if __EMOTI_TTS_DEBUG__:
        print(f"Emotivoice history_modifier: \n{history}")
    
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history

rvcModels = rvc.refresh_model_list()

def rvc_click(gen, output_file, rvc_model):
    print("rvc modelnya ",rvc_model)
    if rvc_model != "" and rvc.model_file != rvc_model:
        print("update rvc")
        onChangeRvcModel(rvc_model)
    rvc.on_button_click(gen,params['rvc_model_index'],str(output_file),params['rvc_pitch'],params['rvc_crepe_hop'],params['rvc_method_entry'],params['rvc_retrieval_rate'])
    
def output_modifier(emotionPrompt: str = "",string: str = "",rvc_model: str = ""):
    """
    This function is applied to the model outputs.
    """
    global params, current_params
    
    def swap_model_for_llm():
        if params['model_swap']:
            unload_model()
            # load_llm()
    
    # Restore processing_message
    print("*Is typing...*")

    base64_encoded = ""
    try:
        refresh_model = False

        for i in params:
            if params[i] != current_params[i]:
                current_params = params.copy()
                break

        if not current_params['activate']:
            if __EMOTI_TTS_DEBUG__:
                print(f"Emotivoice inactive output_modifier: \n{string}")
            return string
        
        if __EMOTI_TTS_DEBUG__:
            print(f"Emotivoice active output_modifier: \n{string}")

        if models is None:
            refresh_model = True

        if params['model_swap']:
            refresh_model = True

        if refresh_model:
            unload_model()
            load_model()

        if models is None:
            print('[Emotivoice_RVC_TTS] No models loaded')
            return string

        original_string = string
        # we don't need to handle numbers. The text normalizer in tortoise does it better
        string = tts_preprocessor.replace_invalid_chars(string)
        string = tts_preprocessor.replace_abbreviations(string)
        string = tts_preprocessor.clean_whitespace(string)

        if __EMOTI_TTS_DEBUG__:
            print(f"Emotivoice output_modifier after tts_preprocessor: \n{string}")

        if string == '':
            string = '*Empty reply, try regenerating*'
            swap_model_for_llm()
            return string

        out_dir_root = params['output_dir'] if params['output_dir'] is not None and Path(params['output_dir']).is_dir() \
            else 'extensions/Emotivoice_RVC_TTS/outputs'

        output_dir = Path(out_dir_root).joinpath('parts')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = Path(out_dir_root).joinpath(f'test_{int(time.time())}.wav')

        if '|' in string:
            texts = string.split('|')
        else:
            texts = split_and_recombine_text(string, desired_length=params['sentence_length'], max_length=1000)

        # Call generate audio and save numpy output by wavio
        gen = generate_audio(emotionPrompt, texts)
        if rvc.model_loaded:
           rvc_click(gen, output_file, rvc_model)
        else:
            if rvcModels:
                indexPath = rvc.selected_model(rvcModels[0])
                params.update({"rvc_model_index": indexPath})
                params.update({"rvc_model": rvcModels[0]})
                if not rvc.model_loaded:
                    print("rvc model load error, using default emotivoice")
                    wavio.write(str(output_file), gen, config.sampling_rate)
                else:
                    rvc_click(gen, output_file)
            else:
                print("no rvc model loaded, using default emotivoice")
                wavio.write(str(output_file), gen, config.sampling_rate)

        print(f"output path {output_file}")
        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

        with open(output_file, 'rb') as wav_file:
            wav_data = wav_file.read()
            base64_encoded = base64.b64encode(wav_data).decode('utf-8')

    except Exception as e:
        string = f'Emotivoice TTS error: \n\n{traceback.format_exc()}'
        string += f'-----------\n\n{original_string}'
        print(e)
        
    
    swap_model_for_llm()
    return base64_encoded


def generate_audio(emotionPrompt, texts):
    # only cat if it's needed
    if len(texts) <= 0:
        return []
    
    print(f"emotion prompt {emotionPrompt}")

    all_parts = []
    for j, text in enumerate(texts):
        # voice tone
        # "8975"
        # params['voice']
        gen = tts(emotionPrompt, text, "8975", models)
        all_parts.append(gen)

    # Emotivoice requires in16 conversion from float
    full_audio = torch.cat(all_parts, dim=-1).numpy().astype('int16')
    return full_audio
    

def setup():
    global params, current_params
    current_params = params.copy()
    if not params['model_swap']:
        load_model()

def onChangeRvcModel(value):
    indexPath = rvc.selected_model(value)
    params.update({"rvc_model_index": indexPath})
    params.update({"rvc_model": value})
    return # gr.Textbox.update(value=indexPath)

def onRefreshModel():
    rvc.refresh_model_list()
    return # gr.Dropdown.update(choices=rvc.model_folders)

def onChangeRvcMethod(value):
    params.update({"rvc_method_entry": value})
    if value == "crepe" or value == "crepe-tiny":
        return # gr.Textbox.update(visible=True)
    return # gr.Textbox.update(visible=False)

# def ui():
#     global controls, params
#     # Gradio elements
#     with # gr.Accordion("Emotivoice RVC TTS"):
#         with # gr.Row():
#             controls['activate'] = # gr.Checkbox(value=params['activate'], label='Activate TTS')
#             controls['autoplay'] = # gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')
#             controls['disable_text_stream'] = # gr.Checkbox(value=params['disable_text_stream'], label='Disable text streaming')

#         controls['prompt'] = # gr.Textbox(value=params['prompt'], label='Emotion prompt for speaker (happy/sad)')
#         controls['show_text'] = # gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
#         controls['voice_dropdown'] = # gr.Dropdown(value=params['voice'], choices=speakers, label='Voice Tone')
#         controls['output_dir_textbox'] = # gr.Textbox(value=params['output_dir'], label='Custom Output Directory')
#         controls['model_swap'] = # gr.Checkbox(value=params['model_swap'], label='Unload LLM Model to save VRAM')
#         controls['sentence_picker'] = # gr.Number(value=params['sentence_length'], precision=0, label='Per Audio Words Slicing (# of words)', interactive=True)
#         controls['rvc_language'] = # gr.Dropdown(value=params['rvc_language'], choices=["english_or_chinese","indonesia"], label='Language')
#         controls['rvc_model'] = # gr.Dropdown(value=params['rvc_model'], choices=rvc.model_folders, label='Rvc Model')
#         controls['rvc_model_index'] = # gr.Textbox(value=params['rvc_model_index'], label='Rvc Model index')
#         controls['refresh_model'] = # gr.Button('Refresh Rvc Model')

#         controls['rvc_method_entry'] = # gr.Dropdown(value=params['rvc_method_entry'], choices=["dio", "pm", "harvest", "crepe", "crepe-tiny" ], label='F0 Method')
#         controls['rvc_pitch'] = # gr.Number(value=params['rvc_pitch'], precision=0, label='Pitch')
#         controls['rvc_crepe_hop'] = # gr.Number(value=params['rvc_crepe_hop'], precision=0, label='Crepe Hop')
#         controls['rvc_retrieval_rate'] = # gr.Number(value=params['rvc_retrieval_rate'], label='Retrieval Rate')
        
#         with # gr.Row():
#             controls['rm_tts_from_hist'] = # gr.Button('Permanently remove generated tts from the all historical message storages')
#             controls['rm_tts_from_hist_cancel'] = # gr.Button('Cancel', visible=False)
#             controls['rm_tts_from_hist_confirm'] = # gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

#         with # gr.Row():
#             controls['cls_output_dir'] = # gr.Button('Clear the tts from custom output directory')
#             controls['cls_output_dir_cancel'] = # gr.Button('Cancel', visible=False)
#             controls['cls_output_dir_confirm'] = # gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

#     controls['refresh_model'].click(lambda: onRefreshModel(),outputs=controls['rvc_model'])
#     # remove tts history with confirmation
#     controls['rm_tts_from_hist_arr'] = [controls['rm_tts_from_hist_confirm'], controls['rm_tts_from_hist'], controls['rm_tts_from_hist_cancel']]
#     controls['rm_tts_from_hist'].click(lambda: [# gr.update(visible=True), # gr.update(visible=False), # gr.update(visible=True)], None, controls['rm_tts_from_hist_arr'])
#     controls['rm_tts_from_hist_confirm'].click(
#             lambda: [# gr.update(visible=False), # gr.update(visible=True), # gr.update(visible=False)], None, controls['rm_tts_from_hist_arr']).then(
#             remove_tts_from_history, gradio('history'), gradio('history')).then(
#             chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
#             chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))
#     controls['rm_tts_from_hist_cancel'].click(lambda: [# gr.update(visible=False), # gr.update(visible=True), # gr.update(visible=False)], None, controls['rm_tts_from_hist_arr'])

#     # clear output dir
#     controls['cls_output_dir_arr'] = [controls['cls_output_dir_confirm'], controls['cls_output_dir'], controls['cls_output_dir_cancel']]
#     controls['cls_output_dir'].click(lambda: [# gr.update(visible=True), # gr.update(visible=False), # gr.update(visible=True)], None, controls['cls_output_dir_arr'])
#     controls['cls_output_dir_confirm'].click(
#             lambda: [# gr.update(visible=False), # gr.update(visible=True), # gr.update(visible=False)], None, controls['cls_output_dir_arr']).then(
#             clear_output_dir, None)
#     controls['cls_output_dir_cancel'].click(lambda: [# gr.update(visible=False), # gr.update(visible=True), # gr.update(visible=False)], None, controls['cls_output_dir_arr'])

#     # Toggle message text in history
#     controls['show_text'].change(
#             lambda x: params.update({"show_text": x}), controls['show_text'], None).then(
#             toggle_text_in_history, gradio('history'), gradio('history')).then(
#             chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
#             chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

#     # Event functions to update the parameters in the backend
#     controls['activate'].change(lambda x: params.update({"activate": x}), controls['activate'], None)
#     controls['autoplay'].change(lambda x: params.update({"autoplay": x}), controls['autoplay'], None)
#     controls['disable_text_stream'].change(lambda x: params.update({"disable_text_stream": x}), controls['disable_text_stream'], None)
#     controls['prompt'].change(lambda x: params.update({"prompt": x}), controls['prompt'], None)
#     controls['voice_dropdown'].change(lambda x: params.update({"voice": x}), controls['voice_dropdown'], None)
#     controls['output_dir_textbox'].change(lambda x: params.update({'output_dir': x}), controls['output_dir_textbox'], None)
#     controls['model_swap'].change(lambda x: params.update({'model_swap': x}), controls['model_swap'], None)
#     controls['sentence_picker'].change(lambda x: params.update({'sentence_length': x}), controls['sentence_picker'], None)
#     controls['rvc_language'].change(lambda x: params.update({"rvc_language": x}), controls['rvc_language'], None)
#     controls['rvc_model'].change(lambda x: onChangeRvcModel(x), controls['rvc_model'], controls['rvc_model_index'])
#     controls['rvc_model_index'].change(lambda x: params.update({"rvc_model_index": x}), controls['rvc_model_index'], None)
#     controls['rvc_method_entry'].change(lambda x: onChangeRvcMethod(x), controls['rvc_method_entry'], controls['rvc_crepe_hop'])
#     controls['rvc_pitch'].change(lambda x: params.update({"rvc_pitch": x}), controls['rvc_pitch'], None)
#     controls['rvc_crepe_hop'].change(lambda x: params.update({"rvc_crepe_hop": x}), controls['rvc_crepe_hop'], None)
#     controls['rvc_retrieval_rate'].change(lambda x: params.update({"rvc_retrieval_rate": x}), controls['rvc_retrieval_rate'], None)

