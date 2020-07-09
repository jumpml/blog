import React from "react"
import Layout from "../components/layout"
import SEO from "../components/seo"
import style from "./about.module.css"

const AboutPage = () => {
  return (
    <Layout>
      <SEO
        title="About JumpML"
        description="Information about JumpML"
        image="/jumpML.svg"
        pathname="/about"
        // Boolean indicating whether this is an article:
        // article
      />
      <section className={style.wrapper}>
        <h1 className={style.heading}>About Us</h1>
        <p>
          Our goal is to help people build useful and intelligent things. We are
          passionate about math, technology and design and we will be sharing
          our knowledge by posting on this site.
        </p>
        <p>
          For customers with very specific algorithm needs, we offer the
          following services
          <ol>
            <li> Customizable training on advanced DSP/ML/SW topics</li>
            <li> DSP/ML and Software consulting services </li>
            <li> Licensing of our ML/DSP algorithms for your projects</li>
          </ol>
          For questions and more details, please contact us at{" "}
          <a href="mailto:jumpml.com@gmail.com">JumpML email.</a>
        </p>
      </section>
    </Layout>
  )
}

export default AboutPage
