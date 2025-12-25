import Markdown from 'react-markdown';
import useAutoScroll from './hooks/useAutoScroll';
import Spinner from './Spinner.jsx';
import { User, AlertCircle } from 'lucide-react';

function ChatMessages({ messages, isLoading }) {
    const scrollContentRef = useAutoScroll(isLoading);

    return (
        <div ref={scrollContentRef} className='grow space-y-4 overflow-y-auto'>
            {messages.map(({ role, content, loading, error }, idx) => (
                <div key={idx} className={`flex items-start gap-4 py-4 px-3 rounded-xl ${role === 'user' ? 'bg-primary-blue/10' : ''}`}>
                    {role === 'user' && (
                        <div className="h-[26px] w-[26px] shrink-0 bg-primary-blue text-white rounded-full flex items-center justify-center">
                            <User size={16} />
                        </div>
                    )}
                    <div className="flex-1">
                        <div className='markdown-container'>
                            {(loading && !content) ? <Spinner />
                                : (role === 'assistant')
                                    ? <Markdown>{content}</Markdown>
                                    : <div className='whitespace-pre-line'>{content}</div>
                            }
                        </div>
                        {error && (
                            <div className={`flex items-center gap-1 text-sm text-error-red ${content && 'mt-2'}`}>
                                <AlertCircle size={20} />
                                <span>Error generating the response</span>
                            </div>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
}

export default ChatMessages;