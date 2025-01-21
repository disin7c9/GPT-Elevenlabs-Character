# from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import ListProperty
from kivy.animation import Animation
from kivy.metrics import dp

# need to update kivmd 1.2.0 => 2.0.0
from kivymd.app import MDApp
from kivymd.uix.button import MDIconButton

import pyaudio
from pydub import AudioSegment
import io
import os
import re
import time
import wave
import numpy as np
import threading
# import whisper
import speech_recognition as sr
from openai import OpenAI
from elevenlabs import Voice, VoiceSettings, play, save
from elevenlabs.client import ElevenLabs


KV = '''
#:import RGBA kivy.utils.rgba

<ImageButton@ButtonBehavior+Image>:
    size_hint: None, None
    size: self.texture_size

    canvas.before:
        PushMatrix
        Scale:
            origin: self.center
            x: .75 if self.state == 'down' else 1
            y: .75 if self.state == 'down' else 1

    canvas.after:
        PopMatrix

BoxLayout:
    orientation: 'vertical'
    padding: dp(5), dp(5)
    RecycleView:
        id: rv
        data: app.messages
        viewclass: 'Message'
        do_scroll_x: False

        RecycleBoxLayout:
            id: box
            orientation: 'vertical'
            size_hint_y: None
            size: self.minimum_size
            default_size_hint: 1, None
            # magic value for the default height of the message
            default_size: 0, 38
            key_size: '_size'

    FloatLayout:
        size_hint_y: None
        height: 0
        Button:
            size_hint_y: None
            height: self.texture_size[1]
            opacity: 0 if not self.height else 1
            text:
                (
                'go to last message'
                if rv.height < box.height and rv.scroll_y > 0 else
                ''
                )
            pos_hint: {'pos': (0, 0)}
            on_release: app.scroll_bottom()

    BoxLayout:
        size_hint: 1, None
        size: self.minimum_size
        TextInput:
            id: ti
            size_hint: 1, None
            height: min(max(self.line_height, self.minimum_height), 30)
            multiline: False
            font_name: 'NanumGothic.ttf'

            on_text_validate:
                app.send_message(self)

        MDIconButton:
            icon: 'send'
            on_release:
                app.send_message(ti)

        MDIconButton:
            icon: 'microphone'
            on_release:
                app.transcript_voice()

<Message@FloatLayout>:
    message_id: -1
    bg_color: '#223344'
    side: 'left'
    text: ''
    size_hint_y: None
    _size: 0, 0
    size: self._size
    text_size: None, None
    opacity: min(1, self._size[0])

    Label:
        text: root.text
        padding: 10, 10
        size_hint: None, 1
        size: self.texture_size
        text_size: root.text_size
        font_name: 'NanumGothic.ttf'

        on_texture_size:
            app.update_message_size(
            root.message_id,
            self.texture_size,
            root.width,
            )

        pos_hint:
            (
            {'x': 0, 'center_y': .5}
            if root.side == 'left' else
            {'right': 1, 'center_y': .5}
            )

        canvas.before:
            Color:
                rgba: RGBA(root.bg_color)
            RoundedRectangle:
                size: self.texture_size
                radius: dp(5), dp(5), dp(5), dp(5)
                pos: self.pos

        canvas.after:
            Color:
            Line:
                rounded_rectangle: self.pos + self.texture_size + [dp(5)]
                width: 1.01
'''

ID_ASSISTANT = '' # ChatGPT assistant ID
ID_VOICE = '' # ElevenLabs voice ID


client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)
assistant = client.beta.assistants.retrieve(
    assistant_id=ID_ASSISTANT
)
thread = client.beta.threads.create()

# transcript_model = whisper.load_model("medium")

client11L = ElevenLabs(
        api_key=os.environ.get("ELEVENLABS_API_KEY")
    )


class MessengerApp(MDApp):
    messages = ListProperty()

    def build(self):
        return Builder.load_string(KV)

    def add_message(self, text, side, color):
        # create a message for the recycleview
        self.messages.append({
            'message_id': len(self.messages),
            'text': text,
            'side': side,
            'bg_color': color,
            'text_size': [None, None],
        })

    def update_message_size(self, message_id, texture_size, max_width):
        # when the label is updated, we want to make sure the displayed size is
        # proper
        if max_width == 0:
            return

        one_line = dp(50)  # a bit of  hack, YMMV

        # if the texture is too big, limit its size
        if texture_size[0] >= max_width * 2 / 3:
            self.messages[message_id] = {
                **self.messages[message_id],
                'text_size': (max_width * 2 / 3, None),
            }

        # if it was limited, but is now too small to be limited, raise the limit
        elif texture_size[0] < max_width * 2 / 3 and \
                texture_size[1] > one_line:
            self.messages[message_id] = {
                **self.messages[message_id],
                'text_size': (max_width * 2 / 3, None),
                '_size': texture_size,
            }

        # just set the size
        else:
            self.messages[message_id] = {
                **self.messages[message_id],
                '_size': texture_size,
            }

    @staticmethod
    def focus_textinput(textinput):
        textinput.focus = True

    def send_message(self, textinput):
        text = textinput.text
        textinput.text = ''
        self.add_message(text, 'right', '#223344')
        self.focus_textinput(textinput)
        Clock.schedule_once(lambda *args: self.answer(text), 1)
        self.scroll_bottom()

    def transcript_voice(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        def listen_and_transcribe():
            with microphone as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=60)
                    print("Finished recording.")
                except sr.WaitTimeoutError:
                    print("Listening timed out while waiting for speaking")
                    return  # Skip the voice input process
                
            # with open("input.wav", "wb") as f:
            #     f.write(audio.get_wav_data())
            
            try:
                # Use the WAV file for transcription
                transcription = recognizer.recognize_google(audio, language="ja")
                print(f"Recognized: {transcription}")
                # Schedule the update on the main thread
                Clock.schedule_once(lambda dt: self.update_text_input(transcription), 0)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

        thread = threading.Thread(target=listen_and_transcribe)
        thread.start()

    def update_text_input(self, transcription):
        self.root.ids.ti.text = transcription
        self.send_message(self.root.ids.ti)
    
    def answer(self, text, *args):
        # query
        query = client.beta.threads.messages.create(
            thread_id=thread.id,
            role='user',
            content=text
        )
        
        # Execute our run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        
        # Wait for completion
        self.wait_on_run(run, thread)
        
        # Retrieve all the messages added after our last user message
        messages = client.beta.threads.messages.list(
            thread_id=thread.id, order="asc", after=query.id
        )

        response = messages.data[0].content[0].text.value
        response = re.sub('【.*?】', '', response)
        
        self.add_message(response, 'left', '#332211')
        self.make_tts(response)
    
    def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    def make_tts(self, text, output:str = 'output.mp3', output_format:str = 'mp3_44100_128'):
        settings=VoiceSettings(
            stability=0.9,  # 감정의 일관성 1 <--> 0 다양성
            similarity_boost=0.9,  # 목소리 유사도 1 <--> 0 닮지 않음 (값이 높으면서, 녹음에 잡소리가 있으면 배경음도 따라하려 할 수 있다)
            style=0, # 원본 화자에 대한 스타일을 강조하고 모방한다. 비교적 최근 도입된 설정. 0값 권장
            use_speaker_boost=True, # 원본 화자에 대한 유사성을 높인다. 비교적 최근 도입된 설정.
        )
        audio = client11L.generate(
            text=text,
            voice=Voice(voice_id=ID_VOICE),
            model='eleven_multilingual_v2',
            output_format=output_format,
            voice_settings=settings,
        )
        save(audio, output)
        self.play_voice_file(output)

    def play_voice_file(self, filename):
        ''' mp3 => wav '''
        
        # Open the MP3 file using pydub
        mp3_audio = AudioSegment.from_mp3(filename)
    
        # Ensure the audio is in stereo
        if mp3_audio.channels == 1:
            mp3_audio = mp3_audio.set_channels(2)
    
        # Convert MP3 to WAV in memory
        wav_io = io.BytesIO()
        mp3_audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Open the WAV file from memory
        wav_file = wave.open(wav_io, 'rb')
    
        # Get audio file properties
        sampwidth = wav_file.getsampwidth()
        nchannels = wav_file.getnchannels()
        framerate = wav_file.getframerate()
    
        # Create a PyAudio instance
        p = pyaudio.PyAudio()
    
        # Open a stream
        try:
            stream = p.open(format=p.get_format_from_width(sampwidth),
                            channels=nchannels,
                            rate=framerate,
                            output_device_index=6, 
                            output=True)
            
            streamTwo = p.open(format=p.get_format_from_width(sampwidth),
                               channels=nchannels,
                               rate=framerate,
                               output_device_index=5, 
                               output=True)
        except OSError as e:
            raise ValueError(f"Failed to open stream: {e}")
    
        # Read data from the file
        data = wav_file.readframes(1024)
    
        # Play the file by streaming the data
        while data:
            stream.write(data)
            streamTwo.write(data)
            data = wav_file.readframes(1024)
    
        # Close the stream and terminate the PyAudio instance
        stream.stop_stream()
        stream.close()
        streamTwo.stop_stream()
        streamTwo.close()
        p.terminate()
        
    def scroll_bottom(self):
        rv = self.root.ids.rv
        box = self.root.ids.box
        if rv.height < box.height:
            Animation.cancel_all(rv, 'scroll_y')
            Animation(scroll_y=0, t='out_quad', d=.5).start(rv)
    

if __name__ == '__main__':
    MessengerApp().run()
