import streamlit as st

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="YouRAG Interactive Player", layout="wide")

st.title("📺 YouRAG: Trò chuyện cùng Video (Local AI)")

# Nhập URL Youtube
video_url = st.text_input("Nhập Link YouTube URL:")

if st.button("Xử lý Video (Ingest)"):
    with st.spinner("Đang tải phụ đề và phân tách Vector..."):
        # res = requests.post(f"{API_URL}/ingest", json={"video_url": video_url})
        st.success("Indexing hoàn tất!")

# Chia bố cục 2 cột (Tính năng ấn tượng nhất)
col_player, col_chat = st.columns([1, 1])

with col_player:
    st.subheader("Trình Phát Video")
    if video_url:
        # Nhúng iFrame trình phát
        # Logic sau khi trả lời AI sẽ "nhảy" đến giây (start_time)
        timestamp = st.session_state.get('jump_timestamp', 0)
        st.video(video_url, start_time=timestamp)

with col_chat:
    st.subheader("Chat với AI")
    user_q = st.chat_input("Hỏi gì đi...")
    
    if user_q:
        st.chat_message("user").write(user_q)
        
        # Quá trình gọi AI
        with st.chat_message("assistant"):
            st.write("Đang truy xuất thông tin liên quan... (RAG stage)")
            # Gọi API
            # res = requests.post(f"{API_URL}/query", json={"question": user_q}).json()
            answer_text = "AI sẽ trả lời ở đây kèm bằng chứng."
            st.write(answer_text)
            
            # Khối hiển thị nguồn
            with st.expander("Nguồn tham khảo (Sources)"):
                # Ví dụ người dùng click vào nút sẽ kích hoạt nhảy Video
                if st.button("Xem video đoạn 02:15"):
                    st.session_state.jump_timestamp = 135 # 2*60 + 15
                    st.rerun()
