import { useEffect, useRef } from 'react';

const useAutosize = (value) => {
    const ref = useRef(null);

    useEffect(() => {
        if (ref.current) {
            ref.current.style.height = 'auto';
            ref.current.style.height = ref.current.scrollHeight + 'px';
        }
    }, [value]);

    return ref;
};

export default useAutosize;
