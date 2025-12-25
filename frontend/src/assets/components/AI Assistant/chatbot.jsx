import { useState, useEffect } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from './chatMessages';
import ChatInput from './chatInput';

import { askQuery } from '../../../services/api';

import { useParams } from 'react-router-dom';

function Chatbot() {
    const { patientId } = useParams();
    const [messages, setMessages] = useImmer([]);
    const [newMessage, setNewMessage] = useState('');

    // Load chat history from sessionStorage on mount
    useEffect(() => {
        if (!patientId) return;

        const storageKey = `chat_history_${patientId}`;
        const savedHistory = sessionStorage.getItem(storageKey);

        if (savedHistory) {
            try {
                const parsedHistory = JSON.parse(savedHistory);
                setMessages(parsedHistory);
                console.log('Loaded chat history:', parsedHistory.length, 'messages');
            } catch (e) {
                console.error('Failed to load chat history:', e);
            }
        }
    }, [patientId]);

    // Save chat history to sessionStorage whenever messages change
    useEffect(() => {
        if (!patientId || messages.length === 0) return;

        const storageKey = `chat_history_${patientId}`;
        sessionStorage.setItem(storageKey, JSON.stringify(messages));
    }, [messages, patientId]);

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
            // Use the centralized API client which handles BASE URL and auth
            const userType = localStorage.getItem('userType') || 'patient';
            const data = await askQuery({
                query: trimmedMessage,
                user_type: userType,
                patient_id: patientId, // Pass patient_id for context-aware answers
                explain: true
            });

            setMessages(draft => {
                const lastMsg = draft[draft.length - 1];
                // The backend returns { answer: "...", ... } or { response: "..." } depending on endpoint
                // query.py returns EnhancedResponse which has 'answer'
                lastMsg.content = data.answer || data.response || "No response received.";
                lastMsg.loading = false;
            });

        } catch (err) {
            console.error(err);
            setMessages(draft => {
                const lastMsg = draft[draft.length - 1];
                lastMsg.loading = false;
                lastMsg.error = true;
                lastMsg.content = "Sorry, something went wrong. Please check your connection.";
            });
        }
    }

    return (
        <div className='relative grow flex flex-col-reverse gap-6 pb-6 mb-5 '>
            <ChatInput
                newMessage={newMessage}
                isLoading={isLoading}
                setNewMessage={setNewMessage}
                submitNewMessage={submitNewMessage}
            />
            <ChatMessages
                messages={messages}
                isLoading={isLoading}
            />
            {messages.length === 0 && (
                <div className='mb-3 font-urbanist text-primary-blue text-xl font-semibold space-y-2 text-center'>
                    <p> Welcome!</p>
                    <p>I am NeuroAI Assistant</p>
                    <p>Ask me about Symptoms, Treatments, or Medical Conditions related to ALzheimer.</p>
                </div>
            )}
        </div>
    );
}

export default Chatbot;