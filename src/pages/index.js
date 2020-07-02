import React from "react"
import Layout from "../components/layout"

import style from "./index.module.css"

const IndexPage = () => {
  return (
    <Layout>
      <section className={style.wrapper}>
        <h1 className={style.heading}>Articles</h1>

        <p>Coming Soon!</p>
        <p>We will have lots of articles on several interesting topics.</p>
      </section>
    </Layout>
  )
}

export default IndexPage
