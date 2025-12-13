import useAutosize from './hooks/useAutosize';
import { Send } from 'lucide-react';

function ChatInput({ newMessage, isLoading, setNewMessage, submitNewMessage }) {
    const textareaRef = useAutosize(newMessage);

    function handleKeyDown(e) {
        if (e.keyCode === 13 && !e.shiftKey && !isLoading) {
            e.preventDefault();
            submitNewMessage();
        }
    }

    return (
        <div className='sticky mb-20 bottom-0 shrink-0 py-4 '>
            <div className='p-1.5 rounded-3xl z-50 font-mono origin-bottom animate-chat duration-400'>
                <div className='pr-0.5 bg-white relative shrink-0 rounded-3xl overflow-hidden transition-all border border-indigo-300/20 focus-within:ring-2 focus-within:ring-indigo-500 focus-within:border-indigo-500'>
                    <textarea
                        className='block w-full max-h-[140px] py-2 px-4 pr-11 bg-white rounded-3xl resize-none placeholder:text-primary-blue placeholder:leading-4 placeholder:-translate-y-1 sm:placeholder:leading-normal sm:placeholder:translate-y-0 focus:outline-none border-0'
                        ref={textareaRef}
                        rows='1'
                        value={newMessage}
                        onChange={e => setNewMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type a message..."
                        disabled={isLoading}
                    />
                    <button
                        type='button'
                        className='absolute top-1/2 -translate-y-1/2 right-3 p-1 rounded-md hover:bg-primary-blue/20 disabled:opacity-50 z-10 pointer-events-auto'
                        onClick={(e) => { e.preventDefault(); submitNewMessage(); }}
                        disabled={isLoading || !newMessage.trim()}
                        aria-label='Send message'
                        title='Send'
                    >
                        <Send size={20} className="text-primary-blue" />
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ChatInput;