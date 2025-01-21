import pyaudio
from pydub import AudioSegment
import io
import os
import re
import time
import whisper
import wave

from openai import OpenAI
from elevenlabs import Voice, VoiceSettings, play, save
from elevenlabs.client import ElevenLabs


ID_ASSISTANT = '' # ChatGPT assistant ID
ID_VOICE = '' # ElevenLabs voice ID


client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)
assistant = client.beta.assistants.retrieve(
    assistant_id=ID_ASSISTANT
)
thread = client.beta.threads.create()

model = whisper.load_model("medium")

client11L = ElevenLabs(
        api_key=os.environ.get("ELEVENLABS_API_KEY")
    )


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def get_response(content):
    query = client.beta.threads.messages.create(
        thread_id=thread.id,
        role='user',
        content=content
    )

    # Execute our run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Wait for completion
    wait_on_run(run, thread)
    # Retrieve all the messages added after our last user message
    messages = client.beta.threads.messages.list(
        thread_id=thread.id, order="asc", after=query.id
    )

    response_text = messages.data[0].content[0].text.value
    clean_text = re.sub('【.*?】', '', response_text)
    return clean_text


def play_voice_file(filename):
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


def make_tts(content, output:str = 'output.mp3', output_format:str = 'mp3_44100_192'):
    settings=VoiceSettings(
        stability=0.9,  # 감정의 일관성 1 <--> 0 다양성
        similarity_boost=0.9,  # 목소리 유사도 1 <--> 0 닮지 않음 (값이 높으면서, 녹음에 잡소리가 있으면 배경음도 따라하려 할 수 있다)
        style=0, # 원본 화자에 대한 스타일을 강조하고 모방한다. 비교적 최근 도입된 설정. 0값 권장
        use_speaker_boost=True, # 원본 화자에 대한 유사성을 높인다. 비교적 최근 도입된 설정.
    )
    audio = client11L.generate(
        text=content,
        voice=Voice(voice_id=ID_VOICE),
        model='eleven_multilingual_v2',
        output_format=output_format,
        voice_settings=settings,
    )
    save(audio, output)
    play_voice_file(output)
    

def transcribe_directly():
    sample_rate = 16000
    bits_per_sample = 16
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    def callback(in_data, frame_count, time_info, status):
        wav_file.writeframes(in_data)
        return (None, pyaudio.paContinue)

    wav_file = wave.open('input.wav', 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(bits_per_sample // 8)
    wav_file.setframerate(sample_rate)

    audio = pyaudio.PyAudio()
    input("Press Enter to start recording...")
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=0,
                        stream_callback=callback,
                       )

    input("Press Enter to stop recording...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wav_file.close()

    transcription = model.transcribe('input.wav', language="ja")
    
    return transcription['text']



if __name__ == "__main__":
    while True:
        content = transcribe_directly()
        print(content)
        
        response = get_response(content)
        print(response)
        make_tts(response)