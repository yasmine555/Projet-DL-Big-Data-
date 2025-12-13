import { useState } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from './chatMessages';
import ChatInput from './chatInput';

function Chatbot() {
    const [messages, setMessages] = useImmer([]);
    const [newMessage, setNewMessage] = useState('');

    const isLoading = messages.length > 0 && messages[messages.length - 1].loading;

    async function submitNewMessage() {
        const trimmedMessage = newMessage.trim();
        if (!trimmedMessage || isLoading) return;

        setMessages(draft => {
            draft.push({ role: 'user', content: trimmedMessage });
            draft.push({ role: 'assistant', content: '', loading: true });
        });
        setNewMessage('');

        try {
            const response = await fetch('http://localhost:8002/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: trimmedMessage }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            setMessages(draft => {
                const lastMsg = draft[draft.length - 1];
                lastMsg.content = data.response;
                lastMsg.loading = false;
            });

        } catch (err) {
            console.error(err);
            setMessages(draft => {
                const lastMsg = draft[draft.length - 1];
                lastMsg.loading = false;
                lastMsg.error = true;
                lastMsg.content = "Sorry, something went wrong.";
            });
        }
    }

    return (
        <div className='relative grow flex flex-col gap-6 pt-6 h-[80vh]'>
            {messages.length === 0 && (
                <div className='mt-3 font-urbanist text-primary-blue text-xl font-semibold space-y-2 text-center'>
                    <p> Welcome!</p>
                    <p>I am NeuroAI Assistant</p>
                    <p>Ask me about Symptoms, Treatments, or Medical Conditions related to ALzheimer.</p>
                </div>
            )}
            <ChatMessages
                messages={messages}
                isLoading={isLoading}
            />
            <ChatInput
                newMessage={newMessage}
                isLoading={isLoading}
                setNewMessage={setNewMessage}
                submitNewMessage={submitNewMessage}
            />
        </div>
    );
}

export default Chatbot;