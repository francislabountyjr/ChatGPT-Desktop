from revChatGPT.V1 import Chatbot


class ChatBot():
    """
    Chatbot class
    """
    def __init__(self, config: dict, conversation_id: str = None, parent_id: str = None) -> None:
        """
        Initialize chatbot
        
        Parameters:
            config (dict): config
            conversation_id (str): conversation id
            parent_id (str): parent id
        """
        self.chatbot = Chatbot(config=config)
        self.conversation_id = conversation_id
        self.parent_id = parent_id

    def get_response(self, prompt: str, conversation_id: str = None, parent_id: str = None) -> str:
        """
        Get response

        Parameters:
            prompt (str): prompt
            conversation_id (str): conversation id (optional)
            parent_id (str): parent id (optional)

        Returns:
            str: response
        """
        if conversation_id is None:
            conversation_id = self.conversation_id
        if parent_id is None:
            parent_id = self.parent_id

        response = self.chatbot.ask(prompt=prompt, conversation_id=conversation_id, parent_id=parent_id)
        for i, partial_response in enumerate(response):
            if i == 0:
                self.conversation_id = partial_response['conversation_id']
                self.parent_id = partial_response['parent_id']
            yield partial_response['message']

    def get_conversations(self, offset: int = 0, limit: int = 20) -> dict:
        """
        Get conversations

        Parameters:
            offset (int): offset
            limit (int): limit

        Returns:
            dict: conversations
        """
        return self.chatbot.get_conversations(offset=offset, limit=limit)
    
    def get_conversation(self, conversation_id: str = None) -> dict:
        """
        Get conversation

        Parameters:
            conversation_id (str): conversation id

        Returns:
            dict: conversation
        """
        if conversation_id is None:
            conversation_id = self.conversation_id
        messages = self.chatbot.get_msg_history(convo_id=conversation_id)
        if messages is not None:
            self.conversation_id = conversation_id
            self.parent_id = None
        return messages
    
    def change_title(self, title: str, conversation_id: str = None) -> dict:
        """
        Change title

        Parameters:
            title (str): title
            conversation_id (str): conversation id

        Returns:
            dict: conversation
        """
        if conversation_id is None:
            conversation_id = self.conversation_id
        return self.chatbot.change_title(convo_id=conversation_id, title=title)
    
    def reset(self) -> None:
        """
        Reset chatbot
        """
        self.conversation_id = None
        self.parent_id = None
        self.chatbot.reset_chat()

    def rollback_conversation(self, num: int = 1, conversation_id: str = None) -> dict:
        """
        Rollback conversation

        Parameters:
            num (int): number of messages to rollback
            conversation_id (str): conversation id

        Returns:
            dict: conversation
        """
        if conversation_id is None:
            conversation_id = self.conversation_id
        self.chatbot.conversation_id = conversation_id
        return self.chatbot.rollback_conversation(num=num)
    
    def delete_conversation(self, conversation_id: str = None) -> None:
        """
        Delete conversation

        Parameters:
            conversation_id (str): conversation id
        """
        if conversation_id is None or conversation_id == self.conversation_id:
            conversation_id = self.conversation_id
            self.conversation_id = None
            self.parent_id = None
        return self.chatbot.delete_conversation(convo_id=conversation_id)
    
    def delete_all_conversations(self) -> None:
        """
        Delete all conversations
        """
        self.conversation_id = None
        self.parent_id = None
        return self.chatbot.clear_conversations()
