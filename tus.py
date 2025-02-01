from moviepy.editor import VideoFileClip, concatenate_videoclips

def concatenate_videos(video_paths, output_path):
    """
    Concatenate multiple videos into a single video file.
    
    Args:
        video_paths (list): List of paths to input video files
        output_path (str): Path where the output video will be saved
    """
    try:
        # Load all video clips
        video_clips = []
        for path in video_paths:
            clip = VideoFileClip(path)
            video_clips.append(clip)
        
        # Concatenate the clips
        final_clip = concatenate_videoclips(video_clips, method="compose")
        
        # Write the output video
        final_clip.write_videofile(output_path, 
                                 codec='libx264', 
                                 audio_codec='aac')
        
        # Close all clips to free up resources
        for clip in video_clips:
            clip.close()
        final_clip.close()
        
        print(f"Successfully created concatenated video: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Clean up any open clips in case of error
        for clip in video_clips:
            clip.close()

# Example usage
if __name__ == "__main__":
    # List of input video paths
    videos = [
        "/Users/arjuntomar/Desktop/GPT/deep_sort/tus.mp4",
        "/Users/arjuntomar/Desktop/GPT/deep_sort/tus2.mp4",
        "/Users/arjuntomar/Desktop/GPT/deep_sort/tus3.mp4",
        "/Users/arjuntomar/Desktop/GPT/deep_sort/tus4.mp4"
    ]
    
    # Output video path
    output = "concatenated_output.mp4"
    
    # Concatenate the videos
    concatenate_videos(videos, output)