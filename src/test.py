# fix_meta_display.py
with open('app.py', 'r') as f:
    content = f.read()

old_meta = '''        with col_m:
            st.markdown("#### Meta Ads")
            st.metric(
                "Expected Revenue",
                f"${pred['meta_revenue_p50']:,.0f}",
                delta="No conversion data",
            )
            st.warning("Meta revenue data missing")'''

new_meta = '''        with col_m:
            st.markdown("#### Meta Ads")
            st.metric(
                "Expected Revenue",
                f"${pred['meta_revenue_p50']:,.0f}",
                delta=f"${pred['meta_revenue_p90'] - pred['meta_revenue_p10']:,.0f} range",
            )
            if meta_budget > 0:
                st.caption(f"ROAS: {pred['meta_revenue_p50']/meta_budget:.2f}x")'''

if old_meta in content:
    content = content.replace(old_meta, new_meta)
    print("Fixed Meta display!")
else:
    print("Pattern not found - checking for partial match...")
    if 'st.warning("Meta revenue data missing")' in content:
        content = content.replace(
            'st.warning("Meta revenue data missing")',
            'if meta_budget > 0:\n                st.caption(f"ROAS: {pred[\"meta_revenue_p50\"]/meta_budget:.2f}x")'
        )
        print("Fixed with partial match")

with open('app.py', 'w') as f:
    f.write(content)
print("Done")