import os
from dotenv import load_dotenv
from slack_bolt import App
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError



class SlackBot:
      def __init__(self):
        load_dotenv()
        self.app = App(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"]
        )
      
      def get_channel_id_by_name(self, channel_name):
        """Finds channel ID from channel name."""
        if not channel_name:
             print("Channel name cannot be empty.")
             return None
        # Remove leading '#' if present
        clean_channel_name = channel_name.lstrip('#')
        try:
            # Using cursor-based pagination for potentially many channels
            cursor = None
            while True:
                result = self.app.client.conversations_list(
                    types="public_channel,private_channel",
                    limit=200, # Fetch in batches
                    cursor=cursor
                )
                if not result['ok']:
                    print(f"Error fetching channels: {result['error']}")
                    return None

                channels = result.get('channels', [])
                for channel in channels:
                    if channel.get('name') == clean_channel_name:
                        print(f"Found channel '{clean_channel_name}' with ID: {channel['id']}")
                        return channel['id']

                cursor = result.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break # No more pages

            print(f"Channel '{clean_channel_name}' not found.")
            return None
        except SlackApiError as e:
            print(f"Slack API error fetching channels: {e.response['error']}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching channels: {e}", exc_info=True)
            return None

      def send_message(self, text: str, channel: str):  
            assert channel is not None, "Channel is required"
            try:
                  resp = self.app.client.chat_postMessage(channel=channel, text=text)
                  print("Message sent:", resp["ts"])
            except Exception as e:
                  print("Error sending message:", e)

      def send_file(self, file_path: str, channel: str, title: str = None, initial_comment: str = None):
        """
        Uploads a file to Slack using files_upload_v2 and shares it.
        """
        if not channel:
            print("Channel name is required to send a file.")
            return None # Indicate failure
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return None # Indicate failure

        channel_id = self.get_channel_id_by_name(channel)
        if not channel_id:
            print(f"Failed to find channel ID for '{channel}'. Cannot send file.")
            return None # Indicate failure

        file_name = title or os.path.basename(file_path)

        try:
            print(f"Uploading and sharing '{file_name}' to channel {channel_id} using files_upload_v2...")
            # Use files_upload_v2 for direct upload and sharing
            response = self.app.client.files_upload_v2(
                channel=channel_id,
                file=file_path,
                filename=file_name,
                title=file_name,
                initial_comment=(initial_comment if initial_comment else f"Here's the file: {file_name}")
            )

            file_info = response.get('file')
            if file_info:
                print(f"File '{file_name}' (ID: {file_info.get('id', 'N/A')}) successfully uploaded and shared to channel {channel_id}.")
                return file_info
            else:
                print(f"files_upload_v2 succeeded but returned no file info. Response: {response}")
                return None

        except SlackApiError as e:
            print(f"Slack API error during file send (files_upload_v2): {e.response['error']}")
            print(f"Full error response: {e.response}")
            return None
        except Exception as e:
            print(f"Unexpected error sending file: {e}", exc_info=True)
            return None