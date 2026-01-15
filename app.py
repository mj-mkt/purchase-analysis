import streamlit as st
import pandas as pd
import io

st.set_page_config(layout="wide", page_title="구매 데이터 분석")

st.title("🛍️ 쇼핑몰 구매 데이터 분석")

# Sidebar - File Upload
st.sidebar.header("1. 데이터 업로드")
uploaded_files = st.sidebar.file_uploader("CSV 파일 선택 (복수 선택 가능)", type=['csv'], accept_multiple_files=True)

@st.cache_data(show_spinner=False)
def load_and_process_data(files):
    all_dfs = []
    
    for file in files:
        # Try reading with utf-8 first, then cp949
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='cp949')
        all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Preprocessing
    if '주문일시' in combined_df.columns:
        combined_df['주문일시'] = pd.to_datetime(combined_df['주문일시'], errors='coerce')
    
    # New/Repeat Logic (Phone based)
    if '주문자 휴대전화' in combined_df.columns and '주문일시' in combined_df.columns:
        # Calculate first purchase date per phone number
        combined_df['최초주문일시'] = combined_df.groupby('주문자 휴대전화')['주문일시'].transform('min')
        
        # Initialize as New Purchase
        combined_df['구매구분'] = '신규 구매'
        
        # Mark as Repeat if order date is strictly after first purchase date
        combined_df.loc[combined_df['주문일시'] > combined_df['최초주문일시'], '구매구분'] = '재구매'
        
    return combined_df

if uploaded_files:
    with st.spinner('데이터 처리 중...'):
        try:
            raw_df = load_and_process_data(uploaded_files)
            
            if raw_df.empty:
                st.error("데이터가 비어있습니다.")
            else:
                # --- Global Data Cleaning (Sidebar) ---
                st.sidebar.header("2. 데이터 전처리 (공통)")
                
                # 1. Duplicate Check
                total_rows_raw = len(raw_df)
                duplicates_count = raw_df.duplicated().sum()
                
                if duplicates_count > 0:
                    st.sidebar.warning(f"⚠️ 중복된 행이 {duplicates_count:,}건 발견되었습니다.")
                    remove_duplicates = st.sidebar.checkbox("중복 데이터 제거 (완전히 동일한 행)", value=False)
                    if remove_duplicates:
                        df = raw_df.drop_duplicates()
                        st.sidebar.success(f"제거 완료! ({len(df):,}건 남음)")
                    else:
                        df = raw_df.copy()
                else:
                    st.sidebar.info("중복된 데이터가 없습니다.")
                    df = raw_df.copy()
                
                # 2. Status Filter (Auto-detect)
                status_cols = [c for c in df.columns if any(x in c for x in ['상태', 'Status'])]
                if status_cols:
                    selected_status_col = status_cols[0] # Pick first match
                    unique_statuses = df[selected_status_col].unique()
                    
                    st.sidebar.subheader("주문 상태 필터")
                    selected_statuses = st.sidebar.multiselect(
                        f"포함할 '{selected_status_col}' 선택",
                        unique_statuses,
                        default=unique_statuses
                    )
                    
                    if selected_statuses:
                        df = df[df[selected_status_col].isin(selected_statuses)]
                
                # --- Tabs ---
                tab1, tab2 = st.tabs(["📊 기본 분석 (신규/재구매)", "🔄 구매 패턴 분석 (이전→이후)"])
                
                # ==========================================
                # TAB 1: Basic Analysis (Existing Logic)
                # ==========================================
                with tab1:
                    # Check required columns
                    required_cols = ['주문일시', '주문자 휴대전화', '상품번호', '주문번호']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"다음 필수 컬럼이 데이터에 없습니다: {', '.join(missing_cols)}")
                    else:
                        # Basic Filters for Tab 1
                        st.subheader("필터 설정")
                        col_f1, col_f2, col_f3 = st.columns(3)
                        
                        # Purchase Type
                        with col_f1:
                            purchase_types = st.multiselect(
                                "구매 구분",
                                ['신규 구매', '재구매'],
                                default=['신규 구매', '재구매'],
                                key='filter_purchase_type'
                            )
                        
                        # Product ID
                        unique_products = sorted(df['상품번호'].astype(str).unique())
                        with col_f2:
                            selected_products = st.multiselect(
                                "상품 번호 (비어있으면 전체)",
                                unique_products,
                                default=[],
                                key='filter_product_basic'
                            )
                        
                        # Date Range
                        valid_dates = df['주문일시'].dropna()
                        if not valid_dates.empty:
                            min_date = valid_dates.min().date()
                            max_date = valid_dates.max().date()
                            with col_f3:
                                start_date = st.date_input("시작일", min_date, key='filter_start_date')
                                end_date = st.date_input("종료일", max_date, key='filter_end_date')
                        else:
                            start_date, end_date = None, None
                        
                        # Apply Filters
                        filtered_df = df.copy()
                        if purchase_types:
                            filtered_df = filtered_df[filtered_df['구매구분'].isin(purchase_types)]
                        if selected_products:
                            filtered_df = filtered_df[filtered_df['상품번호'].astype(str).isin(selected_products)]
                        if start_date and end_date:
                            mask = (filtered_df['주문일시'].dt.date >= start_date) & (filtered_df['주문일시'].dt.date <= end_date)
                            filtered_df = filtered_df[mask]
                        
                        # Prepare for Display (Helper Col)
                        if '주문번호' in filtered_df.columns:
                            filtered_df = filtered_df.sort_values(by=['주문번호', '주문일시'], ascending=[True, True])
                            filtered_df['주문카운트'] = (~filtered_df.duplicated(subset=['주문번호'], keep='first')).astype(int)
                        else:
                            filtered_df['주문카운트'] = 1

                        # Stats
                        st.divider()
                        c1, c2, c3 = st.columns(3)
                        
                        total_orders_metrics = filtered_df['주문번호'].nunique()
                        c1.metric("총 주문 건수", f"{total_orders_metrics:,}건")
                        
                        new_users_metric = filtered_df[filtered_df['구매구분']=='신규 구매']['주문번호'].nunique()
                        repeat_users_metric = filtered_df[filtered_df['구매구분']=='재구매']['주문번호'].nunique()
                        
                        c2.metric("신규 구매 (주문 기준)", f"{new_users_metric:,}건")
                        c3.metric("재구매 (주문 기준)", f"{repeat_users_metric:,}건")
                        
                        # Data Table
                        st.dataframe(filtered_df, use_container_width=True)
                        
                        # Download
                        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("CSV 다운로드", csv, "basic_analysis.csv", "text/csv")

                # ==========================================
                # TAB 2: Sequence Analysis (New Logic)
                # ==========================================
                with tab2:
                    st.info("💡 **이전 구매 제품**을 샀던 사람들이, **이후(두 번째 이상)**에 어떤 제품(**타겟**)을 구매했는지 분석합니다.")
                    
                    col_s1, col_s2 = st.columns(2)
                    
                    with col_s1:
                        source_products = st.multiselect(
                            "1. 이전 구매 제품 선택 (기준)",
                            unique_products if 'unique_products' in locals() else [],
                            key='seq_source'
                        )
                        st.caption("예: 제품 505 (이걸 먼저 산 사람을 찾습니다)")
                        
                    with col_s2:
                        target_products = st.multiselect(
                            "2. 이후 구매 제품 선택 (타겟)",
                            unique_products if 'unique_products' in locals() else [],
                            key='seq_target'
                        )
                        st.caption("비어있으면 **모든 제품**을 대상으로 합니다.")
                        
                    if source_products:
                        with st.spinner("분석 중..."):
                            # Logic:
                            # 1. Users who bought Source
                            # We use FULL df (but respecting global Duplicate removal/Status filter)
                            # to find users who bought Source.
                            
                            # Filter DF for Source Products
                            source_df = df[df['상품번호'].astype(str).isin(source_products)]
                            target_users = source_df['주문자 휴대전화'].unique()
                            
                            if len(target_users) == 0:
                                st.warning("선택한 '이전 구매 제품'의 구매 내역이 없습니다.")
                            else:
                                # 2. Get First Purchase Date of Source per User
                                # User might have bought Source multiple times. We take the MIN date of buying Source.
                                user_source_min_date = source_df.groupby('주문자 휴대전화')['주문일시'].min().reset_index()
                                user_source_min_date.columns = ['주문자 휴대전화', '기준제품_최초구매일']
                                
                                # 3. Join back to full DF to find SUBSEQUENT orders
                                # We treat 'df' as the full history available in the uploaded files.
                                users_history = df[df['주문자 휴대전화'].isin(target_users)].copy()
                                users_history = pd.merge(users_history, user_source_min_date, on='주문자 휴대전화', how='inner')
                                
                                # 4. Filter: Order Date > Source First Date
                                # This ensures it's a "Later" purchase.
                                subsequent_df = users_history[users_history['주문일시'] > users_history['기준제품_최초구매일']]
                                
                                # 5. Filter: Product must be in Target (IF SELECTED)
                                if target_products:
                                    final_seq_df = subsequent_df[subsequent_df['상품번호'].astype(str).isin(target_products)]
                                else:
                                    final_seq_df = subsequent_df
                                
                                # Stats
                                st.divider()
                                
                                count_users_source = len(target_users)
                                count_users_converted = final_seq_df['주문자 휴대전화'].nunique()
                                conversion_rate = (count_users_converted / count_users_source * 100) if count_users_source > 0 else 0
                                
                                m1, m2, m3 = st.columns(3)
                                m1.metric("기준 제품 구매 고객 수", f"{count_users_source:,}명")
                                
                                target_label = "이후 선택 제품 구매 고객 수" if target_products else "이후 재구매 고객 수 (전체)"
                                m2.metric(target_label, f"{count_users_converted:,}명")
                                m3.metric("전환율", f"{conversion_rate:.1f}%")
                                
                                if not final_seq_df.empty:
                                    st.write(f"##### 분석 결과 ({len(final_seq_df):,}건)")
                                    
                                    # Show interesting columns
                                    cols_seq = ['주문일시', '주문자 휴대전화', '상품번호', '주문번호', '기준제품_최초구매일']
                                    # Add other cols just in case
                                    rest_cols = [c for c in final_seq_df.columns if c not in cols_seq]
                                    st.dataframe(final_seq_df[cols_seq + rest_cols], use_container_width=True)
                                    
                                    # Download
                                    csv_seq = final_seq_df.to_csv(index=False).encode('utf-8-sig')
                                    st.download_button("상세 데이터 다운로드", csv_seq, "sequence_analysis_converted.csv", "text/csv")
                                else:
                                    st.warning("해당 조건(이전 -> 이후)에 맞는 구매 내역이 없습니다.")
                    else:
                        st.info("시작하려면 위에서 '이전 제품'을 선택해주세요.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.code(e)

else:
    st.info("👈 사이드바에서 CSV 파일을 업로드해주세요.")
