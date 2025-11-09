"""
Abstract interface for VLM models used in spatial association experiments.
Supports multiple providers: Gemini, GPT-4o, etc.
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class VLMInterface(ABC):
    """Abstract base class for Vision Language Model interfaces."""
    
    @abstractmethod
    def generate_response(self, video_path: str, prompt: str, response_schema: BaseModel) -> str:
        """
        Generate a response from the model given a video and prompt.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt to send to the model
            response_schema: Pydantic schema for structured output
            
        Returns:
            Raw text response from the model (should be valid JSON)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model being used."""
        pass


class GeminiModel(VLMInterface):
    """Google Gemini model interface."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.0):
        """
        Initialize Gemini model.
        
        Args:
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
    
    def generate_response(self, video_path_1: str, prompt: str, response_schema: BaseModel) -> str:
        """Generate response using Gemini model."""
        print(f"Uploading {video_path_1} to Gemini...")
        myfile_1 = self.client.files.upload(file=video_path_1)
        
        # Wait until the files are processed
        while not myfile_1.state or myfile_1.state.name != "ACTIVE":
            print("Waiting for file processing... state:", myfile_1.state)
            time.sleep(3)
            myfile_1 = self.client.files.get(name=myfile_1.name)
        
        # Generate content
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[myfile_1, prompt],
            config={
                "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        )
        
        return resp.text
    
    def get_model_name(self) -> str:
        """Return the Gemini model name."""
        return self.model_name


class GPT4oModel(VLMInterface):
    """OpenAI GPT-4o model interface."""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 4096, num_frames: int = 20):
        """
        Initialize GPT-4o model.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            num_frames: Number of frames to extract. Use -1 for all frames, or positive integer for specific count
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_frames = num_frames
    
    def _extract_frames_from_video(self, video_path: str, num_frames: int = 20) -> list:
        """
        Extract frames from video for GPT-4o processing.
        GPT-4o doesn't support direct video input, so we extract frames.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract evenly from the video. Use -1 for all frames.
            
        Returns:
            List of base64-encoded frames
        """
        import cv2
        import base64
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # If num_frames is -1, extract all frames
        if num_frames == -1 or num_frames >= total_frames:
            frame_indices = list(range(total_frames))
            print(f"Extracting ALL {total_frames} frames from video...")
        else:
            # Calculate frame indices to extract evenly
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            print(f"Extracting {num_frames} frames from {total_frames} total frames...")
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                # Convert to base64
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frames.append(base64_frame)
        
        cap.release()
        print(f"Successfully extracted {len(frames)} frames")
        return frames
    
    def generate_response(self, video_path_1: str, prompt: str, response_schema: BaseModel, num_frames: int = -1, max_retries: int = 10) -> str:
        """Generate response using GPT-4o model."""

        if num_frames == -1:
            num_frames = self.num_frames
        frames_1 = self._extract_frames_from_video(video_path_1, num_frames)
        
        # Build message content with frames
        content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "text",
                "text": "First Video:"
            }
        ]
        
        # Add frames as images
        for frame in frames_1:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": "high"
                }
            })
        
        print(f"Sending {len(frames_1)} frames to GPT-4o...")
        
        # For structured output, we need to use response_format (if available)
        # or parse the JSON from the response

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}  # Enforce JSON output
                )
            
            
                choice = response.choices[0]
                msg = choice.message
                resp_text = msg.content
                finish = getattr(choice, "finish_reason", None)
                refusal = getattr(msg, "refusal", None)
                usage = getattr(response, "usage", None)

                if resp_text is None:
                    print(f"Empty response received from {self.model_name}")
                    print(f"Finish reason: {finish}")
                    print(f"Refusal: {refusal}")
                    print(f"Usage: {usage}")
                    print(f"Response: {response}")

                    print("We will try again by reducing the number of frames by 1")
                    print(f"Number of frames: {self.num_frames-1}")
                    resp_text = self.generate_response(video_path_1, prompt, response_schema, num_frames=self.num_frames-1)

                return resp_text
            except Exception as e:
                # Catches rate limits, connection errors, and anything else
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print(f"Waiting {5*60} seconds before retry...")
                    time.sleep(5*60)
                else:
                    raise  # Re-raise after max retries

       
    
    def get_model_name(self) -> str:
        """Return the GPT-4o model name."""
        return self.model_name


def create_model(provider: str, **kwargs) -> VLMInterface:
    """
    Factory function to create a model instance.
    
    Args:
        provider: Model provider name ("gemini", "gpt4o")
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        VLMInterface instance
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()
    
    if provider == "gemini":
        return GeminiModel(**kwargs)
    elif provider == "gpt4o" or provider == "gpt-4o":
        return GPT4oModel(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: gemini, gpt4o")
