
import React from "react"
// import { graphql } from "gatsby"
import Layout from "../components/layout"
import SEO from "../components/seo"
import PostLink from "../components/post-link"
import style from "./index.module.css"


const IndexPage = (
) => {
  return (
    <Layout>
      <SEO
        title="JumpML. Learn and enjoy!"
        description="JumpML Website"
        image="/jumpML.svg"
        pathname="/"
        // Boolean indicating whether this is an article:
        // article
      />
      {<section className={style.wrapper}>

      
  <div class="announcement-banner">
  
    <p><strong>Welcome!</strong> This site is now dedicated to sharing knowledge, resources, and tutorials focused on embedded machine learning, voice/audio processing, and large language models (LLMs).</p>
    <p>We aim to provide a focused educational experience for developers, engineers, and enthusiasts, featuring blogs, open-source project reviews, video tutorials, and more.</p>
  </div>
      
      </section> }
    </Layout>
  )
}
export default IndexPage
